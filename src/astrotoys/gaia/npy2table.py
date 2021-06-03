#!/usr/bin/env python
#coding=utf-8
"""Copy selected fields from all binary NumPy files in one directory into a single HDF5 table.

Syntax:
  npy2table.py [options] input_path output_hdf5_table

Options:
  -h  print this help message.
  -f  fields.
  -c  compress level (Default: 0).
  -l  compress library (Default: blosc).
  -t  table title (Default: table of selected fields).
  -p  parallel processes (Default: 1).

Examples:
  npy2table.py ./ gaia.h5:/table_full
  copy all fields into gaia.h5:/table_full

  npy2table.py -f "source_id,ra,dec" ./ gaia.h5:/table_simple
  copy specified fields into gaia.h5:/table_simple

"""

import tables
import sys
import os
import numpy as np
from getopt import gnu_getopt
from time import time
from multiprocessing import cpu_count, Queue, Process
from subprocess import run, DEVNULL, PIPE
from os import path
from astrotoys.formats import gdr2_csv_dtype
from numpy.lib.recfunctions import repack_fields

def import_npy(qin, qout, fields):
    """Import NPY file.
"""
    npyfilename = qin.get()
    while npyfilename is not None:
        a = np.memmap(npyfilename, dtype=gdr2_csv_dtype, mode='r')
        if fields is None:
            qout.put(a[:])
        else:
            qout.put(repack_fields(a[fields][:]))
        npyfilename = qin.get()
    return

def export_table(
        qin,
        h5_filename,
        group_name,
        table_name,
        table_title,
        compress_level,
        compress_library,
        expectedrows):
    with tables.open_file(
            h5_filename,
            'a',
            max_blosc_threads=cpu_count(),
            chunk_cache_size = 1024**3
    ) as h5file:
        if compress_level>0:
            filters = tables.Filters(
                complevel = compress_level,
                complib = compress_library)
        else:
            filters = None
        a = qin.get()
        t = 0
        tic = time()
        h5tab = h5file.create_table(
            group_name,
            table_name,
            a.dtype,
            title         = table_title,
            filters       = filters,
            expectedrows  = expectedrows,
            createparents = True
        )
        print(u"Export to {}:{}, compression level {} (method: {}).".format(
            h5_filename,
            path.join(group_name, table_name),
            compress_level,
            compress_library))
        while a is not None:
            h5tab.append(a[:])
            t += a.size
            sys.stdout.write(u"\r  {}/{} ({:.1f}%), {:.1f} seconds remaining...".format(
                t, expectedrows, 100.0*t/expectedrows, (time()-tic)*(expectedrows-t)/t))
            sys.stdout.flush()
            a = qin.get()
    return

def main():
    fields = None
    compress_library = 'blosc'
    compress_level = 0
    table_title = ''
    nworkers = 1
    opts, args = gnu_getopt(sys.argv[1:], 'hf:c:l:t:p:')
    for opt, val in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt == '-f':
            fields = val.split(',')
        elif opt == '-l':
            compress_library = val
        elif opt == '-c':
            compress_level = int(val)
        elif opt == '-t':
            table_title = val
        elif opt == '-p':
            nworkers = int(val)
        else:
            assert False, 'unhandled option'
    source_dirname = path.normpath(path.abspath(args[0]))
    h5_filename, h5table = args[1].split(':')
    group_name, table_name = path.split(h5table)
    result = run([
        'find', source_dirname, '-type', 'f', '-name', '*.npy', '-ls'
    ], check=True, stdout=PIPE)
    npyfiles = []
    npysizes = []
    for line in result.stdout.decode().splitlines():
        cols = line.split()
        npyfiles.append(cols[-1])
        npysizes.append(int(cols[6]))
    nnpys = len(npyfiles)
    npysizes = np.int64(npysizes)
    expectedrows = np.sum(npysizes) // gdr2_csv_dtype.itemsize
    print(u"{} Numpy files found.".format(nnpys))
    print(u"{} entries to be copied.".format(expectedrows))
    q_files = Queue()
    q_array = Queue(2*nworkers+1)
    importers = []
    for i in range(nworkers):
        proc = Process(target=import_npy, args=(q_files, q_array, fields))
        proc.start()
        importers.append(proc)
    exporter = Process(target=export_table, args=(
        q_array,
        path.normpath(path.abspath(h5_filename)),
        group_name,
        table_name,
        table_title,
        compress_level,
        compress_library,
        expectedrows
    ))
    exporter.start()
    for npy in npyfiles:
        q_files.put(npy)
    for i in range(nworkers):
        q_files.put(None)
    for proc in importers:
        proc.join()
    q_array.put(None)
    exporter.join()
    print(u'\nAll NPY files processed.')
    return

if __name__ == '__main__':
    main()
