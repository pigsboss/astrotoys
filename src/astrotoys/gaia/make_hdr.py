#!/usr/bin/env python3
"""Make HDR image.
Syntax:
    make_hdr.py -i input_npy -o output_image [options]
Options:
    -h  print this help message.
    -m  contrast parameter, [0.3, 1]
    -c  chromatic adaptation parameter, [0, 1]
    -a  light adaptation parameter, [0, 1)
    -f  intensity parameter, [-8, 8]
    -s  shrinking factor (for thumbnails) (Default: 1)
"""

import sys
import numpy as np
from getopt import gnu_getopt
from pimms.filters import hdr_reinhard
from os import path
from imageio import imsave

opts, args = gnu_getopt(sys.argv[1:], 'hi:o:m:a:c:f:b:s:')
b=8
m=.7
a=0.
c=1.
f=0.
s=1
for opt,val in opts:
    if opt == '-h':
        print(__doc__)
        sys.exit()
    elif opt == '-i':
        input_npy = path.abspath(path.normpath(path.realpath(val)))
    elif opt == '-o':
        output_image = path.abspath(path.normpath(path.realpath(val)))
    elif opt == '-m':
        m = float(val)
    elif opt == '-a':
        a = float(val)
    elif opt == '-c':
        c = float(val)
    elif opt == '-f':
        f = float(val)
    elif opt == '-b':
        b = int(val)
    elif opt == '-s':
        s = int(val)
cdata = np.memmap(input_npy, mode='r', dtype='double')
nchns = 3
nrows = int((cdata.size / nchns / 2)**.5)
ncols = int(2*nrows)
cdata = cdata.reshape((nrows, ncols, nchns))
sys.stdout.write('procesing...')
sys.stdout.flush()
rgb   = hdr_reinhard(cdata[::-s, ::s, :], m=m, c=c, a=a, f=f)
sys.stdout.write('\rprocesing...finished\n')
sys.stdout.flush()
if b<=8:
    sys.stdout.write('saving 8-bits HDR image to {:s}...'.format(output_image))
    sys.stdout.flush()
    imsave(output_image, np.uint8(255.*rgb+.5))
elif b<=16:
    sys.stdout.write('saving 16-bits HDR image to {:s}...'.format(output_image))
    sys.stdout.flush()
    imsave(output_image, np.uint16(65535.*rgb+.5))
else:
    sys.stdout.write('saving 32-bits HDR image to {:s}...'.format(output_image))
    sys.stdout.flush()
    imsave(output_image, np.uint32(4294967295.*rgb+.5))
sys.stdout.write('\nHDR image saved.\n')
sys.stdout.flush()
