# -*- coding: utf-8 -*-
"""
This module provides the class `Instrument`, which holds informations necessary to identify the radiotelescope in use.
"""
#------------------
# Imports
#------------------
import os
import sys
from astropy import units as u
from astropy.coordinates import EarthLocation
from pytz import timezone
# Special packages
from skyfield import api
from skyfield.api import load
from skyfield.api import Loader
#------------------

class Instrument:
    """This is a radiotelescope."""

    def __init__(self, name=None, lon=None, lat=None, elev=None, timezone=None, Alt=None, Az=None, fwhm=None, verbose=True, path=None):
        """Instantiate and go."""
        self._name = name
        self._path = path
        self._verbose = verbose
        self._timezone = timezone
        self.lon = lon
        self.lat = lat
        self.elev = elev
        self.location = None
        self.observatory = None
        self._Alt = Alt
        self._Az = Az
        self._fwhm = fwhm
        load = Loader('../data/auxiliary/')
        self.eph = load('de440s.bsp')
        self.earth = self.eph['earth']
        return

    @property
    def name(self):
        """Return the name of the instrument."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name of the instrument."""
        self._name = name

    @property
    def path(self):
        """Return the path of the instrument data files."""
        return self._path

    @path.setter
    def path(self, path):
        """Set the path of the instrument data files."""
        self._path = path

    @property
    def verbose(self):
        """Return verbosity level. Boolean."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Set verbosity level. Boolean."""
        self._verbose = verbose

    @property
    def timezone(self):
        """Return verbosity level. Boolean."""
        return self._timezone

    @timezone.setter
    def timezone(self, timezone):
        """Set verbosity level. Boolean."""
        self._timezone = timezone

    @property
    def fwhm(self):
        """FWHM."""
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        """Set FWHM."""
        self._fwhm = fwhm

    @property
    def Alt(self):
        """Return Altitude."""
        return self._Alt

    @Alt.setter
    def Alt(self, Alt):
        """Set Altitude."""
        self._Alt = Alt

    @property
    def Az(self):
        """Return Azimuth."""
        return self._Az

    @Az.setter
    def Az(self, Az):
        """Set Azimuth."""
        self._Az = Az


    def set_location(self, **kwargs):
        """Determine location of experiment as Astropy EarthLocation object. If not specified, values from class instante defined in self.init are used.

        Args:
            lon (Angle unit): longitude `lon`.
            lat (Angle unit): lat `lat`.
            hgt (Distance unit): Elevation on Earth Model WGS84 `hgt`.

        Raises:
            type: TypeError
            Failed EarthLocation

        """
        for key in ['lon', 'lat', 'elev']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        try:
            self.location = EarthLocation(lon=self.lon, lat=self.lat, height=self.elev )
        except Exception as err:
                raise TypeError('Coordinates could not be parsed. \
                                 \n (lat,lon,hgt) expected.')
        return self


    def set_observatory(self, **kwargs):
        """Determine location of experiment as Skyfield object. If not specified, values from class instance defined in self.init are used.

        Args:
            lon (Angle unit): longitude `lon`.
            lat (Angle unit): lat `lat`.

        Raises:
            type: TypeError
            Failed EarthLocation

        """
        for key in ['lon', 'lat', 'elev']:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        try:
            uirapuru_geo = api.wgs84.latlon(self.lat.value, self.lon.value, self.elev.value )
            self.observatory = uirapuru_geo
        except Exception as err:
                raise TypeError('Coordinates could not be parsed. \
                                 \n (lat,lon) expected.')
        return self




def main():
    """Run the main dummy function."""
    message = ''.join([
    '\n\n This is instrument module of uirapuru package.',
    '\n Check Documentation\n'
    ])
    print(message)
    return None

if __name__ == "__main__":
    main()
