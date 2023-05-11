# -*- coding: utf-8 -*-
"""
Este módulo provê instrumentos e backends com  informações padrão.

Instrumento: minihorn.
Controller: linuxbox.
Backend: RTLSDR.
"""
from astropy import units as u
import pytz
import radiotelescope
from radiotelescope.backend.instrument import Instrument
from radiotelescope.backend.controller import LinuxBox
from radiotelescope.backend.gnuradiobackend import GNURadioBackend


# Definindo Instrumento na posiçpão do Uirapuru
lat = -7.211637 * u.deg
lon = -35.908138 * u.deg
elev = 553 * u.m
# Apontamento padrão do Uirapuru
Alt = 84
Az = 0
fwhm = 15
timezone = pytz.timezone("America/Recife")
# Definindo Modos
modes = {"COLD": "01", "WARM": "02", "HOT": "03", "SKY": "59"}

# Instrumento minihorn
minihorn = Instrument(name='Cornetinha', lon=lon, lat=lat,
                                     elev=elev, timezone=timezone,
                                     verbose=True, Alt=Alt, Az=Az, fwhm=fwhm)

# Controller linuxbox
linuxbox = LinuxBox(name="linuxbox", interface="WAN",
                                   user="bingo", remote_port=22,
                                   remote_IP="192.168.15.81",
                                   local_folder="../data/raw/GNURADIO/",
                                   remote_folder="~/SDR/")

# Backend RTLSDRGNU
RTLSDRGNU = GNURadioBackend(controller=linuxbox,
                                           instrument=minihorn,
                                           modes=modes, name="SDR_GNU",
                                           DEVICE="RTL2838")

# Backend RTLSDRPower
# RTLSDRpower = radiotelescope.RTLSDRpowerBackend(controller=linuxbox,
#                                                 instrument=minihorn,
#                                                 modes=modes,
#                                                 name="SDR_POWER")
