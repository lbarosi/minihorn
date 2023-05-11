# -*- coding: utf-8 -*-
"""
Este módulo contém a classe OBSERVATIONS.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 20.10.2022
"""
# General Imports
import logging
# Third Party Imports
import astropy.units as u
from matplotlib.colorbar import Colorbar
from matplotlib import cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import pandas as pd
import scipy as sp
from scipy.constants import c
from scipy.constants import k
import radiotelescope.observations.sky as sky
# Preparando log -----
logger = logging.getLogger(__name__)


class Observations:
    """Class Observations manipulate data already saved."""

    def __init__(self, t_start=None, duration=None, backend=None, path=None):
        """Instantiate and go."""
        self._t_start = t_start
        self._duration = duration
        self.t_end = None
        self._backend = backend
        self._path = path
        self.data = None
        self.sky = None
        return

    @property
    def t_start(self):
        """Return the duration of observation."""
        return self._t_start

    @t_start.setter
    def t_start(self, t_start):
        """Set the initial time of observations."""
        self._t_start = t_start
        return

    @property
    def duration(self):
        """Return the duration of observation."""
        return self._duration

    @duration.setter
    def duration(self, duration):
        """Set the initial time of observations."""
        self._duration = duration
        return

    @property
    def backend(self):
        """Return the object set as backend."""
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set the backend object from backend Class."""
        self._backend = backend
        return

    @property
    def path(self):
        """Return the duration of observation."""
        return self._path

    @path.setter
    def path(self, path):
        """Set the initial time of observations."""
        self._path = path
        return

    def initialize(self):
        """Inicia início e duração se não fornecido em `__init__`.

        Fixa `timezone="UTC"`, início para 00:00h do dia atual e duração de 12 horas.

        Returns:
            type: object Observations.

        """
        if self.backend is None:
            self.timezone = "UTC"
        else:
            self.timezone = self.backend.instrument.timezone
        if self.t_start is None:
            self.t_start = pd.Timestamp.today().tz_localize("UTC").floor("D")
        if self.duration is None:
            self.duration = pd.Timedelta(12, unit="h")
        self.t_end = self.t_start + self._duration
        return self

    def filter_data(self, data, begin=None, freqs=None, duration=None,
                    sampling=None, mode="median"):
        """Filtra o dataframe com os parâmetros chamados.

        Args:
            df (pd.DataFrame): Se não fornecido utiliza `self.data`.
            begin (pd.DateTime): tempo inicial `begin`. Se não fornecido é o início dos dados.
            freqs (list): `[freqs_min, freqs_max]`.
            duration (Quantity): Description of parameter `duration`. Defaults to None.
            sampling (type): Ainda não implementado inteiramente.

        Returns:
            pd.DataFrame: dados filtrados.

        """
        df = data.copy()
        if begin is None:
            begin = df.index[0]
        else:
            begin = begin
        end = df.index[-1]
        if freqs is not None:
            freqmin = freqs[0]
            freqmax = freqs[1]
        else:
            freqmin = df.columns.min()
            freqmax = df.columns.max()
        if duration is not None:
            end = begin + pd.Timedelta(seconds=duration.to(u.s).value)
            # check if it fits inside original interval
            if end > df.index[-1]:
                end = df.index[-1]
        mask = df.columns.where((df.columns >= freqmin) &
                                (df.columns < freqmax)).dropna()
        result = df.loc[begin:end][mask]
        if sampling is not None:
            delta_t = (df.index[-1] - df.index[0]).total_seconds()/df.index.size
            window = int(sampling.to(u.s).value // delta_t)
            if mode=="median":
                result = result.rolling(window, min_periods=1, center=True).median()
            if mode=="mean":
                result = result.rolling(window, min_periods=1, center=True).mean()
        return result

    def load_observation(self, mode="59", extension="fit"):
        """Load observations from local files."""
        # Read filenames and parse timestamps
        filenames = self.backend._get_filenames(extension=extension,
                                                mode=mode).filenames
        filenames = filenames.loc[self.t_start:self.t_end]
        try:
            df = self.backend.load_measurement(filenames=filenames, mode=mode,
                                               extension=extension)
            self.data = df
        except ValueError:
            print("No data found.")
            return None
        return self

    def calibrate(self, data=None, T_rx=None, TEMP=300, T_HOT=300, T_WARM=50, scale=1, flux=True, gain=1):
        if T_rx is None:
            # carrega T_hot
            df_hot = self.backend._get_filenames(extension="fit",
                                                 mode="03").filenames
            df_hot_data = self.backend.load_measurement(filenames=df_hot)
            hot_data = df_hot_data.median(axis=0)
            V_hot = sp.signal.savgol_filter(hot_data,
                                            int(hot_data.shape[0]/50),
                                            2,
                                            mode="nearest")
            # carrega T_warm
            df_warm = self.backend._get_filenames(extension="fit",
                                                  mode="02").filenames
            df_warm_data = self.backend.load_measurement(filenames=df_warm)
            warm_data = df_warm_data.median(axis=0)
            V_warm = sp.signal.savgol_filter(warm_data,
                                             int(warm_data.shape[0]/50),
                                             2,
                                             mode="nearest")
            # temperatura do receptor
            Yc = 10**((V_hot-V_warm) / 10 / scale)
            T_rx = (T_HOT - Yc * T_WARM)/(Yc-1)
        # carrega T_cold
        df_cold = self.backend._get_filenames(extension="fit",
                                              mode="01").filenames
        df_cold_data = self.backend.load_measurement(filenames=df_cold)
        cold_data = df_cold_data.median(axis=0)
        V_cold = sp.signal.savgol_filter(cold_data,
                                         int(cold_data.shape[0]/50),
                                         2,
                                         mode="nearest")
        # Calibrando
        Ys = 10**((data - V_cold) / 10 / scale)
        T_A = T_rx * (Ys - 1) + Ys * TEMP
        result = T_A
        if flux:
            # freqs in fits file are in MHz, need to use Hz here.
            freqs = np.asarray(data.columns)
            bandwidth = (freqs[-1] - freqs[0]) * 1e6
            integration_time = np.abs((data.index[0] -
                                       data.index[1]).total_seconds())
            Aeff = gain * pow(c/(freqs * 1e6), 2) / (4. * np.pi)
            # power in mW
            S_flux = 1000 * (2. * k / np.sqrt(bandwidth * integration_time)) *\
                T_A / Aeff
            # power in dBm
            SdB = 10.0 * np.log10(S_flux)
            result = SdB
        return result, T_rx

    def plot_waterfall(self, df=None, freqs=None, duration=None, sampling=None, colorbar=True, **kwargs):
        freqs = freqs
        duration = duration
        sampling = sampling
        if ((freqs is not None) or (duration is not None) or
                (sampling is not None)):
            # You may filter data inplace.
            df = self.filter_data(df, freqs=freqs, sampling=sampling,
                                  duration=duration)
        # limits
        freqs = df.columns.astype("float")
        begin = df.index[0]
        end = df.index[-1]
        mt = mdates.date2num((begin, end))
        hfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
        fmt_minor = mdates.MinuteLocator(interval=15)
        # average spectrum.
        SN = df.mean(axis=1)/df.std(axis=1)
        spectrum = df.median(axis=0)
        # create grid format.
        fig = plt.figure(figsize=(16, 10))
        grid = plt.GridSpec(10, 9, hspace=0.0, wspace=0.1)
        spectrum_ax = fig.add_subplot(grid[2:-2, :-1])
        hor_fig = fig.add_subplot(grid[0:2, :-1], xticklabels=[],
                                  sharex=spectrum_ax)
        ver_fig = fig.add_subplot(grid[2:-2, -1], sharey=spectrum_ax)
        main = spectrum_ax.imshow(df.T, aspect='auto',
                                  extent=[mt[0], mt[-1], freqs[-1], freqs[0]],
                                  cmap=cm.inferno, **kwargs)
        spectrum_ax.set_ylabel("Frequencies (MHz)")
        spectrum_ax.set_xlabel("Time")
        spectrum_ax.xaxis.set_major_formatter(hfmt)
        if colorbar:
            cbax = fig.add_subplot(grid[9, :-1])
            Colorbar(ax=cbax, mappable=main, orientation="horizontal",
                     ticklocation="bottom")
        # --------------------------
        # plot averaged spectrum in the vertical.
        ver_fig.plot(spectrum, freqs, c='red')
        ver_fig.grid()
        ver_fig.yaxis.tick_right()
        ver_fig.yaxis.set_label_position('right')
        ver_fig.set_xlabel("Flux")
        # plot averaged spectrum in the vertical.
        hor_fig.plot(df.index, SN, c='red')
        hor_fig.set_ylabel("S/N")
        hor_fig.xaxis.tick_top()
        hor_fig.xaxis.set_minor_locator(fmt_minor)
        hor_fig.xaxis.set_major_formatter(hfmt)
        hor_fig.grid()
        return fig

    def plot_ts(self, df=None, begin=None, duration=None, freqs=None):
        if begin is None:
            begin = self.t_start
        if df is None:
            df = self.df_data
        df_filter = self.filter_data(df.loc[begin:], duration=duration,
                                     freqs=freqs)
        fig = self.plot_timeseries(df_filter, self.df_sky)
        return fig

    def make_sky(self, interval=1 * u.min):
        fuso = self.backend.instrument.timezone
        t_start = self.data.index[0].tz_localize(fuso).tz_convert("UTC")
        t_end = self.data.index[-1].tz_localize(fuso).tz_convert("UTC")
        duration = (t_end - t_start).total_seconds() * u.s
        instrument = self.backend.instrument.set_observatory()
        self.sky = sky.Sky(instrument=instrument,
                           t_start=t_start,
                           duration=duration).\
            make_timevector(delta=interval).make_pointings()
        self.sky.get_all_beam()
        return self

# --------------------------------------
# Utilidades para gráficos
# --------------------------------------
def _plot_waterfall(data=None, ax=None, **kwargs):
    freqs = data.columns.astype("float")
    begin = data.index[0]
    end = data.index[-1]
    mt = mdates.date2num((begin, end))
    hfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
    # --------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(data.T, aspect='auto', extent=[mt[0], mt[-1], freqs[-1],
              freqs[0]], cmap=cm.inferno, **kwargs)
    ax.set_ylabel("Frequencies (MHz)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(hfmt)
    return ax


def plot_mosaic(data=None, ax=None, **kwargs):
    axes = []
    num = len(data)
    if (num == 3) or (num > 4):
        cols = 3
    else:
        cols = 2
    rows = int(np.ceil(num / cols))
    dfs = np.empty([rows, cols], dtype="object")
    ii = 0
    if rows > 1:
        for row in np.arange(rows):
            for col in np.arange(cols):
                dfs[row, col] = data[ii]
                ii += 1
    else:
        for col in np.arange(cols):
            dfs[0, col] = data[ii]
            ii += 1
    if ax is None:
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6))
    if rows > 1:
        for row in np.arange(rows):
            for col in np.arange(cols):
                axes.append(_plot_waterfall(data=dfs[row, col],
                                            ax=ax[row, col], **kwargs))
    else:
        for col in np.arange(cols):
            axes.append(_plot_waterfall(data=dfs[0, col],
                                        ax=ax[col], **kwargs))
    return axes


@jit(nopython=True, nogil=True, cache=True, error_model='numpy', fastmath=True,
     boundscheck=False)
def mad_std(data):
    """Calculate Median Absolute Deviation (MAD) for an 1D numpy array.

    Args:
        data (array): numpy 1D `data`.

    Returns:
        array: Array with same shape as `data`.

    """
    # Define function to use as median
    func = np.nanmedian
    data_median = func(data)
    MAD = func(np.abs(data - data_median))
    # MAD equivalent to 1 sigma if normal distribution is assumed
    result = MAD * 1.482602218505602
    return result


def MAD_filter(df, window=20, threshold=3, imputation="max", value=0, axis=0):
    """Apply a MAD filter to dataframe.

    Args:
        df (DataFrame): `df`.
        window (int or tuple): window size to filter, and axis specification `window`. Defaults to (1,20).
        threshold (float): filter values  `threshold` sigma apart. Defaults to 3.
        imputation (str): "max", "median", "constant" indicates how to replace the values `imputation`. Defaults to "max".
        value (float): if `imputation` is "constant", use this value to replace filtered values in dataframe. `value`. Defaults to 0.

    Returns:
        DataFrame: same shape as entry.

    """
    # DataFrame to store filtered data
    if not isinstance(df, pd.DataFrame):
        result = pd.DataFrame(df)
    else:
        result = df.copy(deep=True)
    data_median = result.rolling(window=window,
                                 min_periods=1,
                                 center=True,
                                 closed='both',
                                 axis=axis).apply(np.median,
                                                  engine="numba",
                                                  raw=True).\
        fillna(method='ffill')
    data_MAD = result.rolling(window=window,
                              min_periods=1,
                              center=True,
                              closed='both',
                              axis=axis).\
        apply(mad_std, engine="numba", raw=True).fillna(method='bfill').\
        fillna(method='ffill')

    mask = np.abs(result - data_median) >= threshold*data_MAD

    df_mask = 0 * result
    df_mask[mask] = 1

    if imputation == "max":
        # Substitute filtered values with 3*sigma value.
        result[mask] = threshold * data_median[mask]
    if imputation == "median":
        # Substitute filtered values with median value.
        result[mask] = data_median[mask]
    if imputation == "constant":
        # Substitute filtered values with constant value.
        result[mask] = value

    return result, df_mask


def RFI_filter(df, window=10, threshold=3.5, imputation="median", axis=0):
    """Apply MAD filter and normalize dataframe with parameters.

    Args:
        df (DataFrame): Description of parameter `df`.
        windows (list): list of tuples passed to `normalize`function `windows`. Defaults to [(11,1),(1,5)].
        MAD (tuple): window to apply the MAD filter `MAD`. Defaults to (1,20).
        threshold (float): threshold in number of std `threshold`. Defaults to 3.5.
        norm (str): "orig" or "MAD" indicates the dataframe to be used in the final normalization `norm`. Defaults to "orig".

    Returns:
        DataFrame: Same shape as entry.

    """
    df = df.copy(deep=True)
    df = df.T
    smooth_spectrum, _ = MAD_filter(df.median(axis=1).copy(),
                                    window=window,
                                    threshold=1,
                                    imputation="median",
                                    axis=axis)
    spectrum = (df/smooth_spectrum.values)
    filt_spectrum, mask = MAD_filter(spectrum.copy(),
                                     window=window,
                                     threshold=threshold,
                                     imputation=imputation,
                                     axis=axis)
    filt_spectrum = filt_spectrum.T
    spectrum = spectrum
    mask = mask.T
    return filt_spectrum, spectrum.T, mask


def I_metric(dfF, dfU, axis=0):
    MF = dfF.mean(axis=axis)/dfF.std(axis=axis)
    MU = dfU.mean(axis=axis)/dfU.std(axis=axis)
    result = 10 * np.log10(MF/MU)
    return result
