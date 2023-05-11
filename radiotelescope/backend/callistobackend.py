# -*- coding: utf-8 -*-
"""
Este módulo contém a Classe Backend e seus métodos, representando dispositivo controlador de radiotelescópio, conectado a um objeto da classe Controller, algum tipo de computador.

As classes Backend são RTLSDRpower, CallistoSpectrometer e GNUradio.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 09.04.2022
"""
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
from radiotelescope.backend.backend import Backend as Backend
# Preparando log ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# ------------------
# Implementação de backends: Callisto
# ------------------
class CallistoSpectrometerBackend(Backend):
    """Classe implementa o espectrômetro callisto como backend.

    Fornece métodos para ler os arquivos em formato FIT e PRN.
    Fornece métodos para controlar a operação de gravação de dados.
    """

    def __init__(self, bandwidth=None, controller=None, gain=None,
                 instrument=None, integration_time=None, modes=None,
                 name=None, nominal_slope=None, observing_time=None,
                 temperature=None):
        super().__init__(self, bandwidth=None, controller=None, gain=None,
                         integration_time=None, modes=None, name=None,
                         nominal_slope=None, observing_time=None,
                         temperature=None)

    def load_measurement(self, filenames=None, mode=None, extension="fit"):
        """Implementa método com nome padrão para o carregamento de arquivos.

        Cada implementação de backend pode ter arquivos diferentes.
        Todos tem o mesmo nome como wrapper para método específico do backend.
        """
        if not filenames:
            filenames = self.filenames
        if mode:
            filenames = filenames[filenames["mode"] == mode]
        if extension == "fit":
            files = [
                file for file in filenames if file.split(".")[-1] == extension
                ]
            result = self.fits2df(files)
        elif extension == "prn":
            # To be implemented
            pass
        else:
            print("Method for extension {} not implemented.".format(extension))
            result = None
        return result

    def save_measurement(self):
        print("Método Não implementado para esta classe.")
        pass

    def scan(mode):
        command = "python callisto.py --action start --mode " + str(mode)
        if self.controller.remote:
            self.controller.run_remote(command = command)
        else:
            self.controller.run(command = command)
        return


    def observe(self, duration=None, mode=None, tty="/dev/ttyACM0"):
        """Comanda observação manual de FIT no `mode` indicado por tempo determinado no parâmetro `duration`."""
        total_seconds = utils.parse_time(duration)
        n_meas = np.ceil(total_seconds/(15 * 60))  #cada FIT tem 15min
        # Preparando Callisto
        for ii in np.arange(n_meas):
            self.scan(mode)
        return

    def _from_digits_to_mV(self, df=None):
        """Pretty much what it does."""
        df = df*2500./255.  #floating point to ensure proper broadcasting
        return df

    def _calibrate_slope(self, df_cal):
        last_hot = df.query("mode == 'HOT'").iloc[[-1]]
        last_warm = df.query("mode == 'WARM'").iloc[[-1]]
        hot_data = self.load_measurement(filenames = last_hot.iloc[[0]]["files"].tolist())
        warm_data = self.load_measurement(filenames = last_warm.iloc[[0]]["files"].tolist())
        # factor 10 comes from dB scale.
        self.freqs = np.array(hot_data.median().index)
        self.Dhot = np.array(hot_data.median())
        self.Dwarm = np.array(warm_data.median())
        slope = _from_digits_to_mV(self.Dhot - self.Dwarm)/10.0
        # smoothing slope loess style.
        size = slope.shape[0]
        windows = 10
        slope = savgol_filter(slope, 2 * np.floor(size/2/windows) + 1, 2, mode = "nearest")
        # Calculando Noise Figure
        YcdB = self._from_digits_to_mV(self.Dhot-self.Dwarm)/self.slope
        Yc = pow(10, YcdB/10)
        NF = ENR_warm - 10 * np.log10(Yc-1)
        return NF, slope

    def load_calibration(self, path=None, target_date=None):
        if not target_date:
            target_date = pd.to_datetime(datetime.now())
        df = self._get_filenames(self, path=path, extension="fit", modes={"COLD":"01", "WARM":"02", "HOT": "03"})
        df_cal = df.sort_values("timestamps").groupby("mode").tail(1)
        if df_cal.shape[0] != 3:
            logger.warning("Some calibration modes are missing, run do_calibrate to fix it.")
        elif ((df_cal.timestamp - target_date).components.hours).any() > 24:
            cold_file = df_cal[df_cal.mode == "COLD"].files.iloc[[-1]].tolist()
            self.dcold = self.load_measurement(cold_file)
            self.NF, self.slope = self._calibrate_slope(df_cal)
        else:
            logger.info("Calibration is too old to be used.")
        return

    def do_calibration(self):
        command = "python /usr/localo/bin/callisto.py"
        if self.controller.remote:
            self.controller.run_remote(command = command)
        else:
            self.controller.run(command = command)
        return

    def calibrate(self, data=None, dcold=None):
        if not dcold:
            dcold = self.dcold
        freqs = np.array(data.columns)
        size = freqs.size
        self.Dcold = dcold.median()
        Trx = self.temperature * ( pow(10, self.NF/10.) - 1.)
        Ys = pow(10, self._from_digits_to_mV(data - self.Dcold) / self.slope / 10)
        Trfi = Trx * (Ys - 1) + Ys * self.temperature
        # freqs in fits file are in MHz, need to use Hz here.
        Aeff = self.gain * pow( c/(freqs * 1000000), 2) /(4. * np.pi)
        # power in mW
        S_flux = 1000 * (2. * k_B / np.sqrt(self.bandwidth * self.integration_time)) * Trfi/Aeff
        # power in dBm
        SdB = 10.0 * np.log10(S_flux)
        # Union of DatetimeIndexes
        times = data.index
        df = pd.DataFrame(SdB, columns=freqs, index = times)
        return df
