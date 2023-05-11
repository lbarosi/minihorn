# -*- coding: utf-8 -*-
"""Fornece funções com utilizades para gráficos do céu, equatoriais ou horizontais.

PACKAGE: Radiotelecope
AUTHOR: Luciano Barosi
DATE: 08.05.2022
"""
import numpy as np
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import astropy.units as u
import radiotelescope.misc.utils as utils


def celestial_bbox(sky=None, fwhm=None, duration=None):
    """Calcula bounding box para objeto celeste do tipo astropy."""
    delta_t = utils.parse_time(duration)
    if delta_t > 23 * 60 * 60:  #segundos
        ra_min = 0 * u.deg
        ra_max = 359.99 * u.deg
    else:
        delta = ((delta_t / 3600) * 15/2) * u.deg
        bbox_center = coord.Angle(sky.ra.degree.mean(), unit="deg").wrap_at(180 * u.deg)
        ra_min = coord.Angle(bbox_center - delta - fwhm*u.deg).wrap_at(180*u.deg).degree
        ra_max = coord.Angle(bbox_center + delta - fwhm*u.deg).wrap_at(180*u.deg).degree

    dec_min = sky.dec.degree.min() - fwhm/2
    dec_max = sky.dec.degree.max() + fwhm/2
    top_right = coord.SkyCoord(ra = ra_max, dec = dec_max, unit = 'deg', frame = "icrs")
    bottom_left = coord.SkyCoord(ra = ra_min, dec = dec_min, unit = 'deg', frame = "icrs")
    result = [bottom_left, top_right]
    return result

def pixel_bbox(WCS, coords):
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

def make_axes(sky=None, fwhm=None, duration=None, ra_lim = None, dec_lim = None,  galactic = None, projection = "CAR"):
    """Determine world coordinate axes to use in plot_pointings and set some other nice features."""
    # define bounding box in celestial coordinates
    bbox = celestial_bbox(sky=sky, fwhm=fwhm, duration=duration)
    # Define centro: precisa de degree no final?
    bbox_center = coord.Angle(sky.ra.degree.mean(), unit="deg").wrap_at(180 * u.deg)
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
            bottom_left  = coord.SkyCoord(ra = min(ra_lim), dec = min(dec_lim), frame = "icrs")
            top_right = coord.SkyCoord(ra = max(ra_lim), dec = max(dec_lim), frame = "icrs")
            xmin, ymin = astro.world_to_pixel(bottom_left)
            xmax, ymax = astro.world_to_pixel(top_right)
        else:
            raise ValueError("Both sky (RA, DEC) limits should be set")
    # Create axes
    fig = plt.figure(figsize=(16,9))
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
    ax.coords.grid(color='lightgray', alpha = 0.7, linestyle='solid')
    ax.axhline(yy_sup, color = "skyblue", linewidth = 3)
    ax.axhline(yy_inf, color = "skyblue", linewidth = 3)
    if galactic:
        overlay = ax.get_coords_overlay('galactic');
        overlay.grid(alpha=0.5, linestyle='solid', color='violet');
        overlay[0].set_axislabel('latitude galáctica l');
        overlay[1].set_axislabel('longitude galactica b');
        overlay[0].set_ticks(spacing=15 * u.deg);
        overlay[1].set_ticks(spacing=15 * u.deg);
        overlay[0].set_ticklabel(color = "violet")
        overlay[1].set_ticklabel(color = "violet")
    return ax
