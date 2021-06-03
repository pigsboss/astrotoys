#!/usr/bin/env python
"""Render regional sky map in P3G format.

Syntax:
table2p3g.py INPUT SPECIFIER OUTPUT

INPUT     - HDF5 table contains all sources.
SPECIFIER - An object specifies the destined P3G pixel grid:
            1. a string could be evaluated as a python dict,
            2. a FITS HDU contains a pre-defined P3G pixel grid.
OUTPUT    - Output FITS file.

Example:
table2p3g.py gaia.h5:/minimal "{'phi':0, 'theta':0, 'N':1024}" gaia.fits
Get sources information from table "minimal" contained in HDF5 file gaia.h5,
construct the specified pixel grid, and write P3G pixel grid to FITS file gaia.fits.
"""
import numpy as np
import tables
import sys
import phassdas.sky_map as sm
import astrotoys.photometry as phot
import matplotlib.cm as cm
from phassdas.astro import equ2ga, equ2ec
from pimms.base import imconv, gauss, histnorm
from scipy.misc import imsave
from astropy.io import fits
from time import time
from os import path

BufferSize = 256 * 1024**2 # buffer size in MBytes
h5file, h5obj = sys.argv[1].split(":")
try:
    kwargs = eval(sys.argv[2])
    m = sm.SkyMap(**kwargs)
except:
    try:
        fitsfile,hduidx = sys.argv[2].split(':')
        hduidx = int(hduidx)
    except ValueError:
        fitsfile = sys.argv[2]
        hduidx = 0
    m = sm.SkyMap(fitsfile=fitsfile, HDUidx=hduidx)
m.info()
if m.csys.lower()[:2] == 'eq':
    radec2phitheta = lambda ra,dec: (np.deg2rad(ra),np.deg2rad(dec))
elif m.csys.lower()[:2] == 'ec':
    radec2phitheta = lambda ra,dec: equ2ec(np.deg2rad(ra),np.deg2rad(dec))
elif m.csys.lower[:2] == 'ga':
    radec2phitheta = lambda ra,dec: equ2ga(np.deg2rad(ra),np.deg2rad(dec))
else:
    raise StandardError('Unsupported coordinate system %s'%m.csys)
m.cdata[:] = 0.0
with tables.open_file(h5file, 'r') as h5:
    tab = h5.get_node(h5obj)
    nbuf = BufferSize / tab.rowsize # number of rows in buffer
    tic = time()
    t = 0
    while t<tab.nrows:
        n   = min(nbuf, tab.nrows-t)
        d   = tab.read(t,t+n)
        phi, theta = radec2phitheta(d['ra'], d['dec'])
        u,v,inside = m.grid.locate(phi, theta)
        uidx = np.int64(u[inside])
        vidx = np.int64(v[inside])
        pidx = uidx + vidx*m.grid.NU
        m.cdata += np.reshape(np.bincount(pidx, phot.gmag2photrate(d['phot_g_mean_mag'][inside]),
            m.cdata.size), m.cdata.shape)
        t += n
        sys.stdout.write("    %d/%d (%.1f%%) entries processed, %.1f seconds remaining...\r"%(
            t,tab.nrows,100.0*t/tab.nrows,(time()-tic)*(tab.nrows-t)/t))
        sys.stdout.flush()
    print "\nAll entries processed."
m.save(sys.argv[3], overwrite=True)
print "Sky map saved to %s"%sys.argv[3]
k = gauss((32,32), fwhm=3)
g = np.clip(imconv(np.double(m.cdata), k), 1e-2*(np.min(m.cdata.ravel()[np.argwhere(m.cdata.ravel())])), None)
imsave("%s_preview.png"%path.splitext(sys.argv[3])[0], cm.inferno(histnorm(g)))

