#!/usr/bin/env python
#coding=utf-8
"""hist2d calculate 2D histogram.

Syntax:
  hist2d -x x_axis_col -y y_axis_col -n bins -b buffer_size h5file:/table

Options:
  -x  column for x axis
  -y  column for y axis
  -n  number of bins
  -b  I/O buffer size in MBytes

Copyright: pigsboss@github
"""

import tables
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
from os import path
from multiprocessing import cpu_count, Process, Pipe
from getopt import gnu_getopt

DEFAULT_BINS_NUMBER = 20
DEFAULT_BUFFER_SIZE = 128 ## I/O buffer size in Mbytes


def save_ndarray(array, filename):
    with np.memmap(filename, dtype=array.dtype, mode='w+', shape=array.shape) as fp:
        fp[:] = array[:]

def main():
    opts, args = gnu_getopt(sys.argv[1:], 'hx:y:n:')
    bins = DEFAULT_BINS_NUMBER
    bfsz = DEFAULT_BUFFER_SIZE
    for opt, val in opts:
        if opt=='-x':
            xcol=val
        elif opt=='-y':
            ycol=val
        elif opt=='-n':
            bins=int(val)
        elif opt=='-b':
            bfsz=int(float(val)*1024**2)
        elif opt=='-h':
            print(__doc__)
            sys.exit()
        else:
            assert False, 'unsupported option {}.'.format(opt)
    h5file, h5obj = args[0].split(':')
    H    = np.zeros((bins, bins))
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    with tables.open_file(h5file, mode='r') as f:
        print(u'HDF5 file: {}'.format(path.normpath(path.abspath(h5file))))
        tab = f.get_node(h5obj)
        print(u'Table: {}'.format(h5obj))
        nbuf = bfsz // tab.rowsize
        tic = time()
        t = 0
        sys.stdout.write(u'Overview scanning......')
        sys.stdout.flush()
        while t<tab.nrows:
            n = min(nbuf, tab.nrows-t)
            rows = tab.read(start=t, stop=t+n)
            selected = rows[np.logical_and(np.logical_not(np.isnan(rows[xcol])), np.logical_not(np.isnan(rows[ycol])))]
            if np.size(selected)>0:
                xmin = min(xmin, np.min(selected[xcol]))
                xmax = max(xmax, np.max(selected[xcol]))
                ymin = min(ymin, np.min(selected[ycol]))
                ymax = max(ymax, np.max(selected[ycol]))
            t+=n
            sys.stdout.write(u'\rOverview scanning......{:d}/{:d} ({:6.2f}%)'.format(t, tab.nrows, 100.0*t/tab.nrows))
            sys.stdout.flush()
        sys.stdout.write(u'\rOverview scanning......Finished. ({:.1f} seconds)'.format(time()-tic))
        sys.stdout.flush()
        xedges = np.arange(nbins+1)/nbins*(xmax-xmin)+xmin
        yedges = np.arange(nbins+1)/nbins*(ymax-ymin)+ymin
        sys.stdout.write(u'Calculating......')
        sys.stdout.flush()
        while t<tab.nrows:
            n = min(nbuf, tab.nrows-t)
            rows = tab.read(start=t, stop=t+n)
            selected = rows[np.logical_and(np.logical_not(np.isnan(rows[xcol])), np.logical_not(np.isnan(rows[ycal])))]
            if np.size(selected)>0:
                H+=np.histogram2d(selected[xcol], selected[ycol], (xedges, yedges))[0]
            t+=n
            sys.stdout.write(u'\rCalculating......{:d}/{:d} ({:6.2f}%)'.format(t, tab.nrows, 100.0*t/tab.nrows))
            sys.stdout.flush()
        sys.stdout.write(u'\rCalculating......Finished. ({.1f} seconds)'.format(time()-tic))
        sys.stdout.flush()
    save_ndarray(H,      '{}_H_{}_{}.npy'.format(path.splitext(h5file)[0], xcol, ycol))
    save_ndarray(xedges, '{}_xedges_{}.npy'.format(path.splitext(h5file)[0], xcol))
    save_ndarray(yedges, '{}_yedges_{}.npy'.format(path.splitext(h5file)[0], xcol))
    plt.imshow(H)

if __name__ == '__main__':
    main()
