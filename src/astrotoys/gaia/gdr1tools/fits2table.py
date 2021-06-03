#!/usr/bin/env python
"""Copy selected fields from all FITS files in one directory into a single HDF5 table.

Syntax:
  fits2table.py path fields hdf5_table

Examples:
  fits2table.py . "" gaia.h5:/table_full
  copy all fields into gaia.h5:/table_full

  fits2table.py . "source_id,ra,dec" gaia.h5:/table_simple
  copy specified fields into gaia.h5:/table_simple

"""
import tables
import sys
import os
import numpy as np
from time import time
from multiprocessing import cpu_count
from os import path
from astropy.io import fits

nrowperfits = 218453
fitsfiles = []
print "Processing FITS files in %s"%sys.argv[1]
for f in os.listdir(sys.argv[1]):
    if path.isfile(path.join(sys.argv[1],f)):
        if f.lower()[-5:]=='.fits' or f.lower()[-8:]=='.fits.gz':
            fitsfiles.append(path.join(sys.argv[1], f))
nfits = len(fitsfiles)
print "%d FITS files found."%nfits
print "Approximately %d entries to be copied."%(nrowperfits * nfits)
    
if len(sys.argv[2]) > 0:
    fields = sys.argv[2].split(',')
    tabletitle = 'Table of selected fields'
    print "Selected fields: %s"%sys.argv[2]
else:
    fields = None
    tabletitle = 'Table of all fields'
    print "Selected all fields."

if nfits>0:
    hdulst = fits.open(fitsfiles[0], 'readonly')
    if fields is None:
        rowdtype = hdulst[1].data.dtype
    else:
        rowdtype = np.copy(hdulst[1].data[:1])[fields].dtype
    hdulst.close()
else:
    print __doc__
    raise StandardError("No FITS file to be processed.")

#filters = tables.Filters(complevel=5, complib='blosc')
#ncpu = cpu_count()
#tables.set_blosc_max_threads(ncpu)

ofile, otable = sys.argv[3].split(":")
print "Output HDF5 file: %s"%ofile
print "Destined table: %s"%otable
ogrp,otab = path.split(otable)
h5file = tables.open_file(ofile, 'a')
h5tab  = h5file.create_table(ogrp, otab, rowdtype, title=tabletitle, expectedrows=nrowperfits*nfits, createparents=True)
t = 0
tic = time()
for f in fitsfiles:
    hdulst = fits.open(f, 'readonly')
    if fields is None:
        h5tab.append(hdulst[1].data.T.tolist())
    else:
        h5tab.append(np.copy(hdulst[1].data)[fields].T.tolist())
    t+=1
    sys.stdout.write("  processing %s (%d/%d), %.1f seconds remaining...\r"%(f,t,nfits,(nfits-t)*(time()-tic)/t))
    sys.stdout.flush()
print '\nAll FITS files processed.'
h5file.close()
