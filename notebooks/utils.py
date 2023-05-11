import os
import glob
import sys
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import dask.dataframe as dd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import scipy.signal
import scipy.ndimage
import skyfield.api as api
from skyfield.framelib import galactic_frame
# import scipy as sp
# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.split(os.getcwd())[0])))
sys.path.append(os.path.abspath(os.path.join(os.path.split(os.getcwd())[0], "radiotelescope")))
sys.path.append(os.path.abspath(os.path.join(os.path.split(os.getcwd())[0], "radiotelescope/GNURadio")))
from radiotelescope.observations.observations import Observations as Obs
from radiotelescope.instruments import RTLSDRGNU
from radiotelescope.observations.observations import plot_mosaic
from radiotelescope.instruments import RTLSDRGNU


def load_radiodata(begin=None, duration=None, backend=None, name=None, folder=None, 
                   mode="59"):
    if duration is None:
        duration = pd.Timedelta(24, unit="h")
    if backend is None:
        backend = RTLSDRGNU
    if folder is None:
        folder = "../data/raw/GNURADIO/"
    obs = Obs(t_start=begin, duration=duration).initialize()
    obs.backend = backend
    obs.backend.name = name
    obs.backend.controller.local_folder = folder
    filenames = obs.backend._get_filenames(extension="fit", mode=mode).filenames
    filenames = filenames.loc[obs.t_start:obs.t_end]
    MBsize = filenames.files.apply(lambda row: float(os.path.getsize(row) / 1024 ** 2)).sum()
    print("Dados tem {:.2f} Mb".format(MBsize))
    obs.load_observation(extension="fit", mode=mode)
    obs.make_sky()
    return obs

def baseline_subtract(data, baseline, window_length=71):
    baseline = scipy.signal.savgol_filter(baseline.median(axis=0), 
                                          window_length=window_length, 
                                          polyorder=2, 
                                          mode="nearest")
    if not data.shape[1] == baseline.shape[0]:
        baseline = baseline[::2]
    result = (data - baseline)
    return result, baseline

def plot_baseline(baseline, data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(data.data.columns, baseline)
    ax.set_xlabel("MHz")
    ax.set_ylabel("dB")
    ax.grid()
    return ax
    
def plot_median(data, threshold=None, distance=None, width=None, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,4))
    median = data.median().values
    freqs = data.columns
    ax.plot(freqs, median)
    peaks, _ = scipy.signal.find_peaks(median, threshold=threshold, width=width, distance=distance)
    ax.scatter(freqs[peaks], 1.01 * median[peaks], color="red", marker="x")
    for peak in peaks:
        ax.text(freqs[peak], 1.05 * median[peak], str(np.round(freqs[peak],2)), fontsize=6)
    ax.set_xlabel("MHz")
    ax.set_ylabel("dB")
    ax.grid()
    return ax

    
def altaz2gal(ALT, AZ, TIME=None):
    ts = api.load.timescale()
    if TIME is None:
        tt = ts.now()
    else:
        tt = ts.from_datetime(TIME)
    planets = api.load('de440.bsp')
    earth = planets['earth']
    antenna = earth + RTLSDRGNU.instrument.set_observatory().observatory
    direction = antenna.at(tt).from_altaz(alt_degrees=ALT, az_degrees=AZ)
    glat, glon, gdistance = direction.frame_latlon(galactic_frame)
    return np.round(glat.degrees,1), np.round(glon.degrees,1)

def gal2altaz(LAT, LON, TIME=None):
    ts = api.load.timescale()
    if TIME is None:
        tt = ts.now().to_astropy()
    else:
        tt = ts.from_datetime(TIME).to_astropy()
    lat = RTLSDRGNU.instrument.lat
    lon = RTLSDRGNU.instrument.lon
    elev = RTLSDRGNU.instrument.elev
    antenna = EarthLocation(lat=lat, lon=lon, height=elev)
    gal_obj = SkyCoord(LON, LAT, unit=u.deg, frame="galactic")
    altaz = gal_obj.transform_to(AltAz(obstime=tt, location=antenna))
    return np.round(altaz.alt.degree, 1), np.round(altaz.az.degree, 1)