#!/usr/bin/env python
"""Render a level 12 (4096 sides) healpix sky map.
Pixel values are photon rate per unit area (pts/s/cm2) in G band.
The output HEALPix sky map uses:
  1. nested scheme, and
  2. equatorial coordinate.

Syntax:
table2hpx.py INPUT OUTPUT

INPUT  - HDF5 table contains all sources.
OUTPUT - FITS file contains the output HEALPix pixel grid.

"""
import numpy as np
import tables
import sys
import healpy as hp
import astrotoys.photometry as phot
from time import time
from os import path

BufferSize = 256 * 1024**2 # buffer size in MBytes

h5file, h5obj = sys.argv[1].split(":")
with tables.open_file(h5file, 'r') as h5:
    tab = h5.get_node(h5obj)
    hpxmap = np.zeros(hp.nside2npix(4096))
    nbuf = BufferSize * 8 / tab.rowsize # number of rows in buffer
    tic = time()
    t = 0
    while t<tab.nrows:
        n   = min(nbuf, tab.nrows-t)
        d   = tab.read(t,t+n)
        idx = d['source_id'] / 34359738368
        hpxmap += np.bincount(idx, phot.gmag2photrate(d['phot_g_mean_mag']), hpxmap.size)
        t += n
        sys.stdout.write("    %d/%d (%.1f%%) entries processed, %.1f seconds remaining...\r"%(t,tab.nrows,100.0*t/tab.nrows,(time()-tic)*(tab.nrows-t)/t))
        sys.stdout.flush()
    print "\nAll entries processed."
hp.write_map(sys.argv[2], hpxmap, nest=True, dtype=np.float64, coord='C', overwrite=True)
print "HEALPix sky map saved to %s"%sys.argv[2]
