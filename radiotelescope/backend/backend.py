# -*- coding: utf-8 -*-
"""
Este módulo contém a Classe Backend e seus métodos, representando dispositivo controlador de radiotelescópio, conectado a um objeto da classe Controller, algum tipo de computador.

As classes Backend são RTLSDRpower, CallistoSpectrometer e GNUradio.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
import os
from glob import glob
import logging
import pathlib
import sys

from astropy import units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from paramiko.auth_handler import SSHException
from scipy.signal import savgol_filter as savgol_filter
# ------------------
# local imports
# ------------------
# import radiotelescope.backend.controller as Controller
# import radiotelescope.backend.instrument as Instrument
# import callisto
import radiotelescope.misc.multiprocess as multiprocess
import radiotelescope.misc.utils as utils
# Preparando log ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# -------------------------------------------------------------------

class Backend(ABC):
    """Classe abstrata para implementar os backends.

    Cada backend específico implementa os métodos abstratos necessário.
    """

    def __init__(self, bandwidth=None, controller=None, instrument=None,
                 gain=None, integration_time=None, modes=None, name=None,
                 nominal_slope=None, observing_time=None, temperature=None,
                 **kwargs):
        self._bandwidth = bandwidth
        self._controller = controller
        self._instrument = instrument
        self._gain = gain
        self._integration_time = integration_time
        self._modes = modes
        self._name = name
        self._nominal_slope = nominal_slope
        self._observing_time = observing_time
        self._temperature = temperature
        self.is_connected = False
        self.slope = None
        self.NF = None
        self.freqs = None
        self.filenames = None
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            if ("_" + key) in self.__dict__:
                setattr(self, key, value)
            else:
                raise KeyError(k)
        return

    @property
    def bandwidth(self):
        """Retorna a largura de banda do backend."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        """Ajusta a largura de banda."""
        self._bandwidth = bandwidth

    @property
    def controller(self):
        """Retorna o controlador."""
        return self._controller

    @controller.setter
    def controller(self, controller):
        """Define o controlador."""
        self._controller = controller

    @property
    def instrument(self):
        """Retorna o instrument."""
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        """Define o instrumento."""
        self._instrument = instrument

    @property
    def gain(self):
        """Retorna o ganho."""
        return self._gain

    @gain.setter
    def gain(self, gain):
        """Ajjusta o ganho."""
        self._gain = gain

    @property
    def integration_time(self):
        """Retorna o tempo de integração."""
        return self._integration_time

    @integration_time.setter
    def integration_time(self, integration_time):
        """Ajusta o tempo de integração."""
        self._integration_time = integration_time

    @property
    def modes(self):
        """Retorna os modos válidos."""
        return self._modes

    @modes.setter
    def modes(self, modes):
        """Ajusta o dicionário de modos válidos."""
        self._modes = modes

    @property
    def name(self):
        """Returna o nome do backend."""
        return self._name

    @name.setter
    def name(self, name):
        """Ajusta o nome do backend."""
        self._name = name

    @property
    def nominal_slope(self):
        """Retorna a relação entre dV e V."""
        return self._nominal_slope

    @nominal_slope.setter
    def nominal_slope(self, nominal_slope):
        """Ajusta a relação entre dB e V."""
        self._nominal_slope = nominal_slope

    @property
    def temperature(self):
        """Retorna a remperatura física."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Ajusta a temperatura física."""
        self._temperature = temperature

    @property
    def observing_time(self):
        """Retorna o tempo de observação."""
        return self.__observing_time

    @observing_time.setter
    def observing_time(self, observing_time):
        """Ajusta o tempo de observação."""
        self.__observing_time = observing_time

    def _get_filenames(self, path=None, extension=None, mode=None):
        """Obtem todos os arquivos do diretório.

        Obtem os arquivos de um diretório e ordena Dataframe com informações
        de timestamps e modos de operação.
        """
        if not path:
            path = self.controller.local_folder
        # Se extensão não é informada lê todos os arquivos.
        if not extension:
            extension = "*"
        files = []
        filenames = glob(path + self.name + "_" + "*" + "." + extension)
        df = pd.DataFrame({"files":filenames})
        # Obtem timestamp do nome dos arquivos e
        # junta com `T` para ler no formato isort.
        df['timestamps'] = df.files.apply(lambda row: "T".join(row.split('/')[-1].split('.')[-2].split("_")[-3:-1]))
        df["mode"] = df.files.apply(lambda row: (row.split("/")[-1].split(".")[-2].split("_")[-1]))
        df["mode"] = df["mode"].astype("str")
        if mode is not None:
            df = df[df["mode"] == mode]
        # Índice do dataframe é o tempo.
        # É registrada informação de horário UTC.
        try:
            #df['timestamps'] = pd.to_datetime(df['timestamps'], format="%Y%m%dT%H%M%S", utc=True)
            df['timestamps'] = pd.to_datetime(df['timestamps'], format="%Y%m%dT%H%M%S")
            df["timestamps"] = df["timestamps"].dt.tz_localize(self.instrument.timezone)
            # Ordenamento temporal é fundamental para efetuar as filtragens.
            self.filenames = df.set_index('timestamps').sort_index()
        except (TypeError, ValueError):
            print("Nomes de arquivos não parecem ser válidos. Eles tem informações de timestamp?")
            self.filenames = None
            pass
        return self

    def connect(self):
        """Define variável booleana que informa sobre a possibilidade de realizar conexão com controlador do backend."""
        try:
            self.controller = self.controller.connect()
            self.is_connected = True
        except AttributeError:
            print("Impossível conectar.")
        return self

    def disconnect(self):
        """Define variável `is_connected` para `False` indicando que instrumento está conectado a máquina local."""
        self.is_connected = False
        return self

    @staticmethod
    def fits2df(filenames=None, n_integration=100):
        """Lê arquivo FIT em um dataframe.

        A entrada deve ser uma lista. Cada arquivos FIT deve estar no formato callisto.
        A saída é um dataframe com índice temporal.
        Colunas indicando frequência, em formato pronto para imshow.
        """
        hdu_data = []
        timevector = []
        for file in filenames:
            with fits.open(file) as hdul:
                stamp = pd.to_datetime(hdul[0].header['DATE-OBS'] + "T" + hdul[0].header['TIME-OBS'])
                data = hdul[0].data
                hdu_data.append(data)
                freqs = hdul[1].data[0][1]
                times = hdul[1].data[0][0]
                vector = stamp + pd.to_timedelta(times, unit="s")
                timevector.append(vector)
        if len(filenames) > 1:
            # União de todos os índices temporais.
            times = pd.DatetimeIndex(np.unique(np.hstack(timevector)))
            # Empilhando leituras. Colunas são frequências.
            data = np.vstack(hdu_data)
        else:
            times = pd.DatetimeIndex(np.unique(timevector))
            data = hdu_data[0]
        result = pd.DataFrame(data, columns=freqs, index=times).sort_index()
        return result

    @abstractmethod
    def load_measurement():
        pass

    # @abstractmethod
    # def save_measurement():
    #     pass

    @abstractmethod
    def observe():
        pass

    @abstractmethod
    def calibrate():
        pass
