# -*- coding: utf-8 -*-
"""
Este módulo contém a classe GNURadioBackend e seus métodos, representando dispositivo controlador de radiotelescópio, conectado a um objeto da classe Controller, algum tipo de computador. Este objeto específico tem a captura de dados ocorrendo por meio de um script GNURadio rodando em background.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 26.04.2022
DATE: 23.10.2022
DATE: 25.10.2022



"""
import importlib
import logging
import multiprocessing
# ------------------
# local imports
# ------------------
import radiotelescope.misc.utils as utils
from radiotelescope.backend.backend import Backend as Backend
# Preparando log ----------------------
logger = logging.getLogger(__name__)

# -------------------------------------
# Classe argumento para GNURADIO
class Args:
        """Argumentos para script GNURADIO."""

        def __init__(self, **kwargs):
            for key in kwargs:
                setattr(self, key, kwargs[key])

# ------------------
# Implementação de backends: RTLSDRpower
# ------------------
class GNURadioBackend(Backend):
    """Backend GNURadio conectado remota ou localmente e seus métodos de controle."""

    def __init__(self, bandwidth=None, controller=None, instrument=None, gain=None, integration_time=None, modes=None, name=None, nominal_slope=None, observing_time=None, temperature=None, DEVICE="RTL2838", GNUScript=None, **kwargs):
        self.slope = None
        self.NF = None
        self.freqs = None
        self.filenames = None
        self._GNUScript = GNUScript
        self.RTLSDR = None
        self.DEVICE = DEVICE
        logger.debug("Classe GNURADIOBACKEND inicializada.")
        logger.warn("Para observações conecte o dongle na porta USB local.")
        super().__init__(bandwidth=bandwidth, controller=controller,
                         instrument=instrument, gain=gain, integration_time=integration_time, modes=modes,
                         name=name, nominal_slope=nominal_slope,
                         observing_time=observing_time, temperature=temperature,
                         **kwargs)


    @property
    def GNUScript(self):
        """Propriedade: nome da instância."""
        return self._GNUScript


    @GNUScript.setter
    def GNUScript(self, GNUScript):
        """Propriedade: nome da instância."""
        self._GNUScript = GNUScript


    def load_measurement(self, filenames=None, mode=None, extension="fit"):
        """Implementa método com nome padrão para o carregamento de arquivos.

        Cada implementação de backend pode ter arquivos diferentes.
        Todos tem o mesmo nome como wrapper para método específico do backend.
        """
        if filenames is None:
            filenames = self.backend._get_filenames().filenames
        if mode:
            filenames = filenames[filenames["mode"] == mode]
        if extension == "fit":
            files = [
                file for file in filenames.files.values if file.split(".")[-1] == extension
                ]
            result = self.fits2df(filenames=files)
        elif extension == "prn":
            # To be implemented
            pass
        else:
            print("Method for extension {} not implemented.".format(extension))
            result = None
        return result


    def save_measurement(self):
        """Não implementado para esta classe ainda."""
        pass


    def observe(self, **kwargs):
        DAEMON = kwargs.pop("daemon", None)
        """Testa conexão com RTL dongle e roda script gnuradio com os parâmetros da observação fornecidos como dicionário."""
        try:
            logger.debug("Importando módulo {}".format(self.GNUScript))
            GNURADIO = importlib.import_module(self.GNUScript)
        except ImportError:
            logger.error("Script GNURADIO não pode ser carregada.")
            return
        args = Args(**kwargs)
        logger.debug("Testando conexão com dispositivo")
        if self.controller.reset_device(self.DEVICE):
            if DAEMON is not True:
                logger.debug("Iniciando GNUradio script local.")
                GNURADIO.main(args)
            else:
                logger.debug("Iniciando GNUradio script local como daemon.")
                multiprocessing.Process(target=GNURADIO.main, args=(args,), daemon=True).start()
        return


    def calibrate(self, data=None, dcold=None):
        """Não Implementado ainda."""
        pass
