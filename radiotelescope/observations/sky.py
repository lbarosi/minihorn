# -*- coding: utf-8 -*-
"""
Este módulo contém a classe SKY

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 07.05.2022
"""
import numpy as np
import pandas as pd
import itertools
import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.time import Time, TimeDelta
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import dask.array as da
from matplotlib.colorbar import Colorbar
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skyfield import api
from skyfield.api import Loader

# --------------------
# Local objects to keep track during observations.
# --------------------
LOCAL_OBJECTS = ['sun', 'moon', 'mercury barycenter', 'venus barycenter',
                 'mars barycenter', 'jupiter barycenter',
                 'saturn barycenter']
# --------------------
# Very strong radiosources
# --------------------
RADIOSOURCES = ['Cassiopeia A', 'Centaurus A', 'Cygnus A', 'Fornax A',
                'Hercules A', 'Hydra A', 'Pictor A', 'Puppis A',
                'Sagittarius A*', 'Taurus A', 'Virgo A']
# --------------------
# GNSS satellite constelation TLE data: orbits
# Constelações principais: GPS, GALILEO, GLONASS, BEIDOU, QZS
# Constelação IRIDIUM opera fora da banda
# --------------------
TLE_urls = ['https://celestrak.com/NORAD/elements/gps-ops.txt',
            'https://celestrak.com/NORAD/elements/glo-ops.txt',
            'https://celestrak.com/NORAD/elements/galileo.txt',
            'https://celestrak.com/NORAD/elements/beidou.txt',
            'https://celestrak.com/NORAD/elements/gnss.txt']
# --------------------
# plate_carre WCS
# --------------------
WCS_PLATE = WCS(naxis=2)
WCS_PLATE.wcs.crpix = [0., 0.]
WCS_PLATE.wcs.cdelt = [1, 1]
WCS_PLATE.wcs.crval = [180, 0]
WCS_PLATE.wcs.ctype = ["RA---CAR", "DEC--CAR"]
# --------------------
# Mollweide WCS
# --------------------
WCS_MOL = WCS(naxis=2)
WCS_MOL.wcs.crpix = [0., 0.]
WCS_MOL.wcs.cdelt = [1, 1]
WCS_MOL.wcs.crval = [180, 0]
WCS_MOL.wcs.ctype = ["RA---MOL", "DEC--MOL"]


class Sky:

    def __init__(self, ephemeris="de440s.bsp", t_start=None, duration=None,
                 FWHM=None, instrument=None, local_objects=LOCAL_OBJECTS):
        """Instantiate and go."""
        self._ephemeris = ephemeris
        self._load = Loader('../data/auxiliary/')
        self._eph = self._load(self.ephemeris)
        self._earth = self._eph['earth']
        self._instrument = instrument
        self._t_start = t_start
        self._duration = duration
        self._FWHM = FWHM
        self.timevector = None
        self.pointings = None
        self.local_objects = local_objects
        self._ts = api.load.timescale()
        return

    @property
    def ephemeris(self):
        return self._ephemeris

    @ephemeris.setter
    def ephemeris(self, ephemeris):
        self._ephemeris = ephemeris

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t_start):
        self._t_start = t_start

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @property
    def local_objects(self):
        return self._local_objects

    @local_objects .setter
    def local_objects(self, local_objects):
        self._local_objects = local_objects

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    def _celestial_bbox(self):
        """Calculate bounding box for astropy coord.Skycord object in sky coordinates."""
        skycoords = self.pointings
        fwhm = self.instrument.fwhm
        bbox_center = coord.Angle(np.ceil(
            self.pointings[self.timevector.shape[0]//2].ra.hour)*15,
            unit="deg").wrap_at(180*u.deg)
        if self.duration > 23 * u.h:
            ra_min = 0 * u.deg
            ra_max = 359.99 * u.deg
        else:
            delta = (self.duration.to(u.h).value * 15/2)*u.deg
            ra_min = coord.Angle(bbox_center - delta - fwhm*u.deg).\
                wrap_at(180*u.deg).degree
            ra_max = coord.Angle(bbox_center + delta).wrap_at(180*u.deg).degree
        dec_min = skycoords.dec.degree.min() - fwhm/2
        dec_max = skycoords.dec.degree.max() + fwhm/2
        top_right = coord.SkyCoord(ra=ra_max, dec=dec_max, unit='deg',
                                   frame="icrs")
        bottom_left = coord.SkyCoord(ra=ra_min, dec=dec_min, unit='deg',
                                     frame="icrs")
        corners = [bottom_left, top_right]
        return corners

    def _make_axes(self, ra_lim=None, dec_lim=None,  galactic=None,
                   projection="CAR"):
        """Determine world coordinate axes to use in plot_pointings and set some other nice features."""
        # define bounding box in celestial coordinates
        bbox = self._celestial_bbox()
        # define central hourangle
        bbox_center = coord.Angle(np.ceil(self.pointings[
            self.timevector.shape[0]//2].ra.hour)*15,
            unit="deg").wrap_at(180*u.deg).degree
        # define world coordinate system
        astro = WCS(naxis=2)
        astro.wcs.crpix = [0., 0.]
        astro.wcs.cdelt = [1, 1]
        astro.wcs.crval = [bbox_center, 0]
        astro.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
        # define pixel limits
        bottom_left = bbox[0]
        top_right = bbox[1]
        xmin, ymin = astro.world_to_pixel(bottom_left)
        xmax, ymax = astro.world_to_pixel(top_right)
        yy_sup = ymax
        yy_inf = ymin
        if ra_lim is not None:
            if dec_lim is not None:
                bottom_left = coord.SkyCoord(ra=min(ra_lim),
                                             dec=min(dec_lim),
                                             frame="icrs")
                top_right = coord.SkyCoord(ra=max(ra_lim),
                                           dec=max(dec_lim),
                                           frame="icrs")
                xmin, ymin = astro.world_to_pixel(bottom_left)
                xmax, ymax = astro.world_to_pixel(top_right)
            else:
                raise ValueError("Both sky (RA, DEC) limits should be set")
        # Create axes
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection=astro)
        lon = ax.coords['RA']
        lat = ax.coords['DEC']
        lon.set_axislabel(r'$\alpha$ (h) - ascenção reta')
        lat.set_axislabel(r'$\delta (^\circ)$ - Declinação')
        lon.set_major_formatter('hh:mm')
        lat.set_major_formatter('dd:mm')
        lon.set_ticks(spacing=2. * u.hourangle)
        lat.set_ticks(spacing=5 * u.deg)
        lon.set_ticks_position('b')
        lon.set_ticklabel_position('b')
        lat.set_ticks_position('l')
        lat.set_ticklabel_position('l')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        ax.invert_xaxis()
        ax.coords.grid(color='lightgray', alpha=0.7, linestyle='solid')
        ax.axhline(yy_sup, color="skyblue", linewidth=3)
        ax.axhline(yy_inf, color="skyblue", linewidth=3)
        if galactic:
            overlay = ax.get_coords_overlay('galactic')
            overlay.grid(alpha=0.5, linestyle='solid', color='violet')
            overlay[0].set_axislabel('latitude galáctica l')
            overlay[1].set_axislabel('longitude galactica b')
            overlay[0].set_ticks(spacing=15 * u.deg)
            overlay[1].set_ticks(spacing=15 * u.deg)
            overlay[0].set_ticklabel(color="violet")
            overlay[1].set_ticklabel(color="violet")
        return ax

    def _pixel_bbox(self, WCS, coords):
        """Return tuple for xlim and ylim in pixel integer coordinates, given a pair of celestial points `coords` and a `WCS`."""
        (xmin, ymax) = WCS.world_to_pixel(coords[0])
        (xmax, ymin) = WCS.world_to_pixel(coords[-1])
        # Pixel coordinates should be integer, round safe.
        xmin = int(np.floor(xmin))
        ymin = int(np.floor(ymin))
        xmax = int(np.ceil(xmax))
        ymax = int(np.ceil(ymax))
        xlim = [xmax, xmin]
        ylim = [ymin, ymax]
        return xlim, ylim

    def beam_on_sky(self):
        """Collect all celestials in one dataframe."""
        pulsares_csv = load_pulsares()
        radiosources_csv = load_radiosources()
        nvss_csv = load_nvss_catalog()
        df_celestials = pd.concat(
            [pulsares_csv.query("DEC >-20 & DEC < 10 & S1400>10")
                [["PSRJ", "RA", "DEC"]].rename(columns={"PSRJ": "NAME"}),
                nvss_csv.query("DEC >-20 & DEC < 10 & S1400>10000")
                [["NVSS", "RA", "DEC"]].rename(columns={"NVSS": "NAME"}),
                radiosources_csv[["SOURCE", "RA", "DEC"]].
                rename(columns={"SOURCE": "NAME"})])
        df = self.get_star_cone(objects=df_celestials)
        df_local_objects = self.get_local_objects_cone()
        df_gnss_satellites = self.get_satellites()
        df = pd.concat([df, df_local_objects, df_gnss_satellites])
        df["TIME"] = pd.to_datetime(Time(df.TIME.values,
                                         format='jd',
                                         scale="tai").to_datetime())
        df.set_index('TIME', inplace=True)
        df = df.sort_index()
        df.reset_index(inplace=True)
        return df

    def make_timevector(self, duration=None, delta=1 * u.h, inplace=True):
        """Create time vector for observations.

        Args:
            duration (quantity): duration of observation `duration`.
            delta (quantity): Interval between observations (Optional) `delta`. Defaults to 1 u.h.
            inplace (boolean): indicates if returns self or timevector.

        Returns:
            type: self or time_vector in skyfield timescale format.

        """
        if duration is None:
            duration = self.duration
        delta = delta.to(u.s).value
        steps = duration.to(u.s).value / delta
        # Astropy timevector is fast to create.
        timelist = Time(self.t_start, scale='utc') +\
            np.arange(steps)*TimeDelta(delta, format='sec', scale='tai')
        # Convert to skyfield timescale for later use.
        timevector = self._ts.from_astropy(timelist)
        if inplace:
            self.timevector = timevector
        else:
            return timevector
        return self

    def _get_altaz_from_radec(self, observer, df):
        """Get AltAz to use with pandas apply."""
        try:
            name = df["NAME"]
        except KeyError:
            name = None
        try:
            obj_type = df["TYPE"]
        except KeyError:
            obj_type = None
        ra = (df["RA"]*u.deg).to(u.hourangle).value
        dec = df["DEC"]
        timestamp = self._ts.tai_jd(df["TIME"])
        celestial = api.Star(ra_hours=ra, dec_degrees=dec)
        pos = observer.at(timestamp).observe(celestial)
        alt, az, _ = pos.apparent().altaz()
        return obj_type, name, timestamp.tai, alt.degrees, az.degrees

    def get_altaz_from_radec(self, observer=None, objects=None):
        df = objects.apply(lambda row:
                           self._get_altaz_from_radec(observer, row),
                           axis=1,
                           result_type='expand')
        if df.shape[1] == 3:
            df.columns = ["TIME", "ALT", "AZ"]
        if df.shape[1] == 4:
            df.columns = ["NAME", "TIME", "ALT", "AZ"]
        if df.shape[1] == 5:
            df.columns = ["TYPE", "NAME", "TIME", "ALT", "AZ"]
        return df

    def _get_radec_target(self, observer=None, objects=None,
                          timevector=None):
        """Get coordinates of objects and returns astropy Skycoord."""
        _ra, _dec, _ = observer.at(timevector).observe(objects).\
            radec(self._ts.J2000)
        coords = coord.SkyCoord(ra=_ra.hours, dec=_dec.degrees,
                                unit=(u.hourangle, u.deg), frame='icrs',
                                equinox='J2000')
        return coords

    def _pos(self, obj, observer, timevector):
        pos = (obj - observer).at(timevector)
        return pos

    def _get_satellites(self, obj, observer, timevector, Alt, Az):
        pos = self._pos(obj, observer, timevector)
        ra, dec, dist = pos.radec()
        cone = observer.at(timevector).from_altaz(alt_degrees=Alt,
                                                  az_degrees=Az).\
            separation_from(pos)
        RA = coord.Longitude(ra._degrees, u.deg,
                             wrap_angle=180*u.deg).to(u.deg).value
        sats = da.from_array([timevector.tai,
                              RA,
                              dec.degrees,
                              cone.degrees,
                              [obj.name] * len(timevector)])
        return sats

    def _get_satellites_df(self, obj, observer, timevector, Alt, Az):
        pos = self._pos(obj, observer, timevector)
        ra, dec, dist = pos.radec()
        cone = observer.at(timevector).from_altaz(alt_degrees=Alt,
                                                  az_degrees=Az).\
            separation_from(pos)
        RA = coord.Longitude(ra._degrees, u.deg,
                             wrap_angle=180*u.deg).to(u.deg).value
        sats = pd.DataFrame({"TIME": timevector.tai,
                             "RA": RA,
                             "DEC": dec.degrees,
                             "ANGLE": cone.degrees,
                             "NAME": [obj.name] * len(timevector)
                             })
        return sats

    def get_satellites(self, timevector=None, observer=None, Alt=None,
                       Az=None, reload=False):

        if not timevector:
            timevector = self.timevector
        if not observer:
            observer = self.instrument.observatory
        if not Alt:
            Alt = self.instrument.Alt
        if not Az:
            Az = self.instrument.Az
        # --------------------
        # Loading satellite data.
        # --------------------
        TLE = TLE_urls
        satellites = []
        for url in TLE:
            satellites.append(self._load.tle_file(url, reload=reload))
        gnss_all = list(itertools.chain(*satellites))
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        # TLE are geocentric, observer should also be geocentric.
        objects = []
        for ii, satellite in enumerate(gnss_all):
            sat_obj = self._get_satellites_df(satellite, observer,
                                              timevector, Alt, Az)
            objects.append(sat_obj)

        df = pd.concat(objects)
        df['TIME'] = df['TIME'].astype(float)
        df['RA'] = df['RA'].astype(float)
        df['DEC'] = df['DEC'].astype(float)
        df['ANGLE'] = df['ANGLE'].astype(float)
        df['NAME'] = df['NAME'].astype(str)

        return df

    def get_local_objects(self, objects=None, CONE=True):
        """Get positions of local objects during observation."""
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        if self.instrument is not None:
            timevector = self.timevector
            fwhm = self.instrument.fwhm
            observer = self._earth + self.instrument.observatory
            if objects is None:
                objects = self.local_objects
            object_list = []
            for sky_object in objects:
                pos = observer.at(timevector).observe(self._eph[sky_object])
                ra, dec, dist = pos.radec()
                cone = observer.at(timevector).from_altaz(
                    alt_degrees=self.instrument.Alt,
                    az_degrees=self.instrument.Az).separation_from(pos)
                df = pd.DataFrame(zip(timevector.tai, ra._degrees,
                                      dec.degrees, cone.degrees, dist.km),
                                  columns=['TIME', 'RA', 'DEC', 'ANGLE',
                                           'DISTANCE'])
                df['NAME'] = [sky_object.split(" ")[0]] * len(timevector)
                object_list.append(df)
            objects_df = pd.concat(object_list)
            if CONE:
                df = objects_df[objects_df.ANGLE < fwhm/2]
        else:
            print("Instrument not set")
            df = None
        return df


    def get_all_beam(self, query_string_nvss="S1400>10000",
                     query_string_psr="S1400>10", cone=True):
        """Collect information from all celestials of interest."""
        df_01 = self.get_local_objects()
        df_02 = self.get_satellites()
        df_03 = self.get_star_cone(
                                   load_nvss_catalog().query(
                                        query_string_nvss))
        df_04 = self.get_star_cone(load_pulsares().query(query_string_psr))
        df_05 = self.get_star_cone(load_radiosources())
        df_list = [df_01, df_02, df_03, df_04, df_05]
        list_objects = ["LOCAL", "GNSS", "NVSS", "PSR", "RADIOA"]
        dfcat = []
        for ii, item in enumerate(df_list):
            if not item.empty:
                item["TYPE"] = list_objects[ii]
                dfcat.append(item)
        df = pd.concat(dfcat)
        if cone:
            df = df[df.ANGLE < self.instrument.fwhm/2]
        return df



    def get_star_cone(self, objects=None, CONE=True):
        """Populate dataframe with distant objects in beam."""
        # --------------------
        # Generate positions in dataframe.
        # --------------------
        instrument_ok = self.instrument is not None
        df_ok = not objects.empty
        if instrument_ok and df_ok:
            timevector = self.timevector
            fwhm = self.instrument.fwhm
            observer = self._earth + self.instrument.observatory
            object_list = []
            for index, star in objects.iterrows():
                ra = (star.RA*u.deg).to(u.hourangle).value
                dec = star.DEC
                celestial = api.Star(ra_hours=ra, dec_degrees=dec)
                pos = observer.at(timevector).observe(celestial)
                ra, dec, dist = pos.radec()
                cone = observer.at(timevector).from_altaz(
                    alt_degrees=self.instrument.Alt,
                    az_degrees=self.instrument.Az).separation_from(pos)
                df = pd.DataFrame(zip(timevector.tai, ra._degrees,
                                      dec.degrees, cone.degrees, dist.km),
                                  columns=['TIME', 'RA', 'DEC', 'ANGLE',
                                           'DISTANCE'])
                df['NAME'] = [star.NAME] * len(timevector)
                object_list.append(df)
            objects_df = pd.concat(object_list)
            if CONE:
                df = objects_df[objects_df.ANGLE < fwhm/2]
        else:
            df = pd.DataFrame()
        return df

    def make_pointings(self, timevector=None, inplace=True, *args, **kwargs):
        """Determine property pointings as a coord.Skycoord object with RA DEC coordinates for the observation times.

        Args:
            timevector (type): Description of parameter `timevector`. Defaults to None.
            *args (type): Description of parameter `*args`.
            **kwargs (type): Description of parameter `**kwargs`.

        Returns:
            type: self

        """
        if self.instrument is not None:
            if timevector is None:
                timevector = self.timevector
            observer = self.instrument.observatory
            ra, dec, _ = observer.at(timevector).from_altaz(
                alt_degrees=self.instrument.Alt,
                az_degrees=self.instrument.Az).radec(self._ts.J2000)
            pointings_sky = coord.SkyCoord(ra.hours, dec.degrees,
                                           unit=(u.hourangle, u.deg),
                                           frame='icrs', equinox='J2000')
            self.pointings = pointings_sky
        else:
            print("Instrument not set")
        if not inplace:
            return None
        return self

    def make_pointings_df(self, interval=None, utc=True):
        """Construct dataframe with times and pointings in human readable format."""
        ra = self.pointings.ra.degree
        dec = self.pointings.dec.degree
        name = [self.instrument.name] * len(self.timevector)
        if utc:
            time = pd.to_datetime(self.timevector.utc_datetime())
        else:
            time = pd.to_datetime(self.timevector.utc_datetime(),
                                  utc=True).\
                                    tz_convert(self.instrument.timezone)
        df = pd.DataFrame({"RA": ra, "DEC": dec, "NAME": name}, index=time)
        if interval is not None:
            df = df.asfreq(freq=interval, method='bfill')
        return df

    def plot_pointings(self, timestamps=True, circles=True, utc=True,
                       interval="1h", ra_lim=None, dec_lim=None,
                       h_offset=7, v_offset=-10, legend_offset=-.1,
                       galactic=True, wcs="CAR"):
        """Set the stage for observations."""
        if self.duration < 24 * u.h:
            sky = self.make_pointings_df(interval=interval).\
                reset_index().rename(columns={"index": "TIME"})
            fwhm = self.instrument.fwhm
            # create world coordinate axes
            ax = self._make_axes(ra_lim=ra_lim, dec_lim=dec_lim,
                                 galactic=galactic, projection=wcs)
            # plot artists
            for ii, row in sky.iterrows():
                if timestamps:
                    if utc:
                        time_text = row.TIME.strftime("%H:%M") + "-UT"
                    else:
                        time_text = row.TIME.astimezone(
                            self.instrument.timezone).\
                                strftime("%H:%M") + "-local"
                    ax.text(row.RA + h_offset, row.DEC + v_offset,
                            time_text, transform=ax.get_transform("world"),
                            color="sienna")
                if circles:
                    c = Circle((row.RA, row.DEC), fwhm/2,
                               transform=ax.get_transform('world'),
                               edgecolor="sienna", facecolor="None")
                    ax.add_patch(c)
            # show pointings as red points.
            ax.scatter(x=sky.RA, y=sky.DEC, marker="+", color="red",
                       transform=ax.get_transform('world'), label="pointing")
            # make room for large legend.
            ax.legend(loc='lower center',
                      bbox_to_anchor=(0.5, legend_offset),
                      ncol=7, fancybox=True, shadow=True)
            artist = ax
        else:
            print("Plotting observation longer than 24hs in the sky does not yield good results. Try another approach.")
            return None
        return artist

    def plot_sky(self, objects=None, markersize=15):
        """Plot AtzAz sky."""
        if objects is None:
            objects = self.get_all_beam()
            objects = self.get_altaz_from_radec(objects=objects)
            objects["TIME"] = pd.to_datetime(objects.TIME.values,
                                             unit="D", origin="julian")
        else:
            objects = objects
        objects["AZwrap"] = coord.Angle(objects.AZ,
                                        unit="deg").wrap_at(180*u.deg).degree
        begin = objects.TIME.min().strftime("%D - %H:%M")
        end = objects.TIME.max().strftime("%H:%M")
        fig, ax = plt.subplots(figsize=(16, 9))
        for celestial in objects.NAME.unique():
            df = objects[objects.NAME == celestial]
            ax.scatter(x=df.AZwrap, y=df.ALT, label=celestial,
                       s=markersize, marker="*")
        ax.grid()
        ax.set_xlim(-90, 90)
        ax.set_ylim(60, 90)
        ax.set_xlabel("Azimute Az")
        ax.set_ylabel("Altitude Alt")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=7,
                  title="Observation started at {} and finished at {}".
                  format(begin, end))
        return ax

    def plot_timeseries(self, df_data, interval="1h"):
        """Plot waterfall and upper panel with celestials."""
        df_sky = self.get_all_beam()
        df_sky["TIME"] = pd.to_datetime(df_sky.TIME.values,
                                        unit="D", origin="julian")
        df_sky["TIME"] = df_sky["TIME"].dt.tz_localize("UTC").dt.tz_convert(self.instrument.timezone)
        df_fit = df_data.copy()
        df_fit = df_fit.reset_index()
        df_fit["index"] = df_fit["index"].dt.\
            tz_localize(self.instrument.timezone)
        df_fit = df_fit.set_index("index")
        # Set up the axes with gridspec
        freqs = df_fit.columns
        begin = df_fit.index[0]
        end = df_fit.index[-1]
        ymin = self.pointings.dec.min().degree - self.instrument.fwhm
        ymax = self.pointings.dec.max().degree + self.instrument.fwhm
        fmt_major = mdates.MinuteLocator(interval=30)
        fmt_minor = mdates.MinuteLocator(interval=15)
        # Set up the axes with gridspec
        mt = mdates.date2num((begin, end))
        hfmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S', tz=self.instrument.timezone)
        # create grid format.
        #fig = plt.figure(figsize=(16, 8))
        fig = plt.figure(figsize=(16, 10))
        #grid = plt.GridSpec(5, 8, hspace=0.0, wspace=0.1)
        grid = plt.GridSpec(10, 9, hspace=0.0, wspace=0.1)
        #spectrum_ax = fig.add_subplot(grid[1:, :-1])
        spectrum_ax = fig.add_subplot(grid[2:-2, :-1])
        sky_ax = fig.add_subplot(grid[0:2, :-1], xticklabels=[],
                                 sharex=spectrum_ax)
        ver_fig = fig.add_subplot(grid[2:-2, -1], sharey=spectrum_ax)
        # waterfall.
        main = spectrum_ax.imshow(df_fit.T, aspect='auto',
                           extent=[mt[0], mt[-1], freqs[-1], freqs[0]],
                           cmap=cm.inferno)
        spectrum_ax.set_ylabel("Frequencies (MHz)")
        spectrum_ax.set_xlabel("Time (local time)")
        spectrum_ax.minorticks_on()
        spectrum_ax.xaxis.set_major_formatter(hfmt)
        spectrum_ax.xaxis.set_minor_locator(fmt_minor)
        spectrum_ax.xaxis.set_tick_params(which='minor', bottom=True)
        # colorbar
        cbax = fig.add_subplot(grid[9, :-1])
        Colorbar(ax=cbax, mappable=main, orientation="horizontal",
                 ticklocation="bottom")
        # SKY
        if not df_sky.empty:
            mask = (df_sky["TIME"] > begin) & (df_sky["TIME"] < end)
            df_sky_filt = df_sky.loc[mask]
            for celeste in df_sky_filt.NAME.unique():
                sky = df_sky_filt[df_sky_filt.NAME == celeste]
                sky_ax.scatter(x=sky.TIME, y=sky.DEC, label=celeste,
                               marker="*")
        sky_ax.axhline(ymin, color="gold", linewidth=2)
        sky_ax.axhline(ymax, color="gold", linewidth=2)
        sky_ax.set_ylabel("Declination")
        sky_ax.xaxis.tick_top()
        sky_ax.xaxis.set_minor_locator(fmt_minor)
        sky_ax.xaxis.set_major_formatter(hfmt)
        sky_ax.grid()
        sky_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 2.0),
                      ncol=7, fancybox=True, shadow=True)
        # Vertical Plot
        spectrum = df_fit.median(axis=0)
        # plot averaged spectrum in the vertical.
        ver_fig.plot(spectrum, freqs, c='red')
        ver_fig.grid()
        ver_fig.yaxis.tick_right()
        ver_fig.yaxis.set_label_position('right')
        ver_fig.set_xlabel("digits")

        return fig


def fetch_nvss_catalogs(filename="../data/auxiliary/nvss_radiosources.csv",
                        DEC_FILTER="<10 && >-30",
                        S1400_mjy=">100",
                        query="DEC >-20 & DEC < 10 & S1400>100"):
    """Fetch astroquery vizier nvss catalog.

    Args:
        DEC_FILTER (type): Filter for Vizier `DEC_FILTER`. Defaults to "<10 && >-30".
        S1400_mjy (type): Filter for Vizier `S1400_mjy`. Defaults to ">10".

    Returns:
        DataFrame

    """
    nvss = "VIII/65/nvss"
    catalog = Vizier(catalog=nvss,
                     columns=['*', '_RAJ2000', '_DEJ2000', 'S1400'],
                     column_filters={"_DEJ2000": "<10 && >-30",
                                     "S1.4": S1400_mjy},
                     row_limit=-1).query_constraints()[nvss]
    df = catalog.to_pandas()[['_RAJ2000', '_DEJ2000', 'NVSS', 'S1.4']]
    df.columns = ['RA', 'DEC', 'NAME', 'S1400']
    df = df[["NAME", "RA", "DEC", "S1400"]]
    df = df.query(query)
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df


def fetch_pulsar_catalogs(filename="../data/auxiliary/pulsares.csv",
                          query="DEC >-20 & DEC < 10 & S1400>10"):
    """Baixa catálogo de pulsares B/psr/psr.

    Baixa catálogo e salva em formato csv no computador local.
    """
    pulsar_table = "B/psr/psr"
    catalog = Vizier(catalog=pulsar_table,
                     columns=['*', 'RA2000', 'DE2000', 'S1400'],
                     row_limit=-1)
    pulsares = catalog.query_constraints()[pulsar_table]
    df = pulsares.to_pandas()[['PSRJ', 'RA2000', 'DE2000', 'Dist', 'P0',
                               'DM', 'S1400']]
    df.dropna(subset=['Dist', 'S1400'], inplace=True)
    # Filtrando fluxos maiores do que 1mJy
    df.columns = ['NAME', 'RA', 'DEC', 'DIST', 'P0', 'DM', 'S1400']
    df = df.query(query)
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df


def fetch_radiosources(filename="../data/auxiliary/radiosources.csv",
                       radiosources=RADIOSOURCES):
    """Baixa radiofontes selecionadas.

    Args:
        radiosources (type): Description of parameter `radiosources`. Defaults to RADIOSOURCES.

    Returns:
        type: Description of returned object.

    """
    s = Simbad()
    radio = s.query_objects(radiosources)
    df = radio.to_pandas()[["MAIN_ID", "RA", "DEC"]]
    df["NAME"] = radiosources
    df["RA"] = coord.Angle(df.RA, unit="hour").degree
    df["DEC"] = coord.Angle(df.DEC, unit="deg").degree
    try:
        df.to_csv(filename, encoding="utf-8", index=False)
        print("arquivo salvo em disco: {}".format(filename))
    except IOError as err:
        print(err + "\n arquivo não foi salvo em disco")
    return df


def load_nvss_catalog(filename="../data/auxiliary/nvss_radiosources.csv",
                      **kwargs):
    """Load previously saved data from NVSS catalog. If file does not exist, fetch data from vizier with kwargs.

    Args:
        filename (type): Defaults to "../data/auxiliary/nvss_radiosources.csv".
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("sources not found on local disk, trying Vizier...")
        df = fetch_nvss_catalogs(**kwargs)
    return df


def load_pulsares(filename="../data/auxiliary/pulsares.csv", **kwargs):
    """Load previously saved data from pulsar catalog. If file does not exist, fetch data from vizier with kwargs.

    Args:
        filename (type): Defaults to "../data/auxiliary/pulsares.csv".
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("sources not found on local disk, trying Vizier...")
        df = fetch_pulsar_catalogs(**kwargs)
    return df

def load_radiosources(filename="../data/auxiliary/radiosources.csv"):
    """Load selected sources. Fetch if file not found.

    Args:
        filename (type): Description of parameter `filename`. Defaults to "../data/auxiliary/radiosources.csv".

    Returns:
        type: Description of returned object.

    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("sources not found on local disk, trying SIMBAD...")
        df = fetch_radiosources(radiosources=RADIOSOURCES)
    return df


def get_galactic_equator(size=720):
    """Return data to plot galactic plane.

    Args:
        size (type): Number of points to plot `size`. Defaults to 720.

    Returns:
        type: Skycoord object.

    """
    ll = np.linspace(0, 360, size)
    bb = np.zeros(size)
    gal_plane = coord.SkyCoord(ll, bb, unit=u.deg, frame="galactic")
    return gal_plane


def plot_df(df, **kwargs):
    """Given a dataframe with RA and DEC information, plots the data with options given..

    Args:
        df (type): dataframe with RA and DEC columns, at least `df`.
        texts (string): keyword indicating name of dataframe columns with texts information to be plotted inside the plot.
        h_offset: use with texts
        v_offset: use with texts
        groups (string): keyword indicating columns of dataframe to use to aggregate the plots. Label will appear in the legend.
        **kwargs (type): Parameters passed directly to matplotlib `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    ax = kwargs.pop("ax", None)
    h_offset = kwargs.pop("h_offset", 0)
    v_offset = kwargs.pop("v_offset", 0)
    coords = coord.SkyCoord(ra=df.RA, dec=df.DEC, unit="deg",
                            frame="icrs")

    if "texts" in kwargs.keys():
        texts = df[kwargs.pop("texts")]
    else:
        texts = None
    if not ax:
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection=WCS_PLATE)
        lon = ax.coords['RA']
        lat = ax.coords['DEC']
        lon.set_axislabel(r'$\alpha$ (h) - ascenção reta')
        lat.set_axislabel(r'$\delta (^\circ)$ - Declinação')
        lon.set_major_formatter('hh:mm')
        lat.set_major_formatter('dd:mm')
        lon.set_ticks(spacing=2. * u.hourangle)
        lat.set_ticks(spacing=5 * u.deg)
        lon.set_ticks_position('bt')
        lon.set_ticklabel_position('bt')
        lat.set_ticks_position('lr')
        lat.set_ticklabel_position('lr')
        ax.set_ylim([-25, 10])
        ax.set_xlim([-180, 180])
        ax.invert_xaxis()
        ax.coords.grid(color='lightgray', alpha=0.7, linestyle='solid')
    if "group" in kwargs.keys():
        group = kwargs.pop('group')
        values = df[group].unique()
        for item in values:
            df_filtered = df[df[group] == item]
            coords = coord.SkyCoord(ra=df_filtered.RA, dec=df_filtered.DEC,
                                    unit="deg", frame="icrs")
            ax.scatter(x=coords.ra, y=coords.dec, **kwargs,
                       transform=ax.get_transform('world'), label=item)
    else:
        ax.scatter(x=coords.ra, y=coords.dec, **kwargs,
                   transform=ax.get_transform('world'))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=7,
              fancybox=True, shadow=True)
    if texts is not None:
        for ii, item in enumerate(coords):
            #matplotlib text shoulb be one by one
            ax.text(item.ra.degree + h_offset, item.dec.degree + v_offset,
                    texts.iloc[ii], transform=ax.get_transform('world'))
    artist = ax
    return artist
