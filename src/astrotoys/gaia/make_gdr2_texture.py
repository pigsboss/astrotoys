#!/usr/bin/env python3
"""Make GDR2 all-sky texture for 3D rendering.

Syntax: make_gdr2_texture.py [options]

Options:
  -h    print this help message
  -i    input (source) HDF5 table path
  -o    output (destination) Numpy memmap path
  -s    Y-size, in pixels (default: 8192)
  -c    coordinate system
"""

import sys
from getopt import gnu_getopt
from os import path
from astrotoys.gaia.map_source_hpx_cl import *

xsize = 16384
ysize =  8192

opts, args = gnu_getopt(sys.argv[1:], 'i:o:s:c:h')

for opt,val in opts:
    if opt == '-i':
        input_path = val
    elif opt == '-o':
        output_path = path.abspath(path.normpath(path.realpath(val)))
    elif opt == '-r':
        ysize = int(val)
        xsize = int(2*ysize)
    elif opt == '-c':
        coord = val

rgb = np.memmap(output_path, shape=(ysize, xsize, 3), dtype='double', mode='w+')
m = HPXMapper(input_path, phot_band='r')
cdata = m.run()
rmap = hp.cartview(cdata, nest=True, coord=coord, return_projected_map=True, xsize=xsize)
m = HPXMapper(input_path, phot_band='g')
cdata = m.run()
gmap = hp.cartview(cdata, nest=True, coord=coord, return_projected_map=True, xsize=xsize)
m = HPXMapper(input_path, phot_band='b')
cdata = m.run()
bmap = hp.cartview(cdata, nest=True, coord=coord, return_projected_map=True, xsize=xsize)
vmax = np.max([rmap, gmap, bmap])
rgb[:,:,0] = rmap/vmax
rgb[:,:,1] = gmap/vmax
rgb[:,:,2] = bmap/vmax
