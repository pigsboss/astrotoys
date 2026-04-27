#!/usr/bin/env python3
#coding=utf-8
"""Convert GDR2's gzipped human-readable CSV (coma separated values) file into binary NumPy array file.

Syntax:
  csv2npy.py [options] csv_file npy_file

Options:
  -d  debug mode.
  -v  verbose.
  -3  Gaia DR3 mode (default is DR2).

Examples:
  csv2npy.py gaiasource0001.csv.gz gaiasource.npy
  convert gaiasource0001.csv.gz to gaiasource.npy

  csv2npy.py -3 gaiadr3.csv.gz dr3.npy
  convert a Gaia DR3 CSV file to npy.
"""
import time
import sys
import gzip
import os
import numpy as np
from getopt import gnu_getopt
from os import path
from astrotoys.formats import gdr2_csv_dtype, gdr3_csv_dtype
import csv
from io import StringIO

def load_gdr2_csv(input_file, verbose=False, debug=False):
    tic=time.time()
    d = {}
    with gzip.open(input_file, 'rt') as f:
        reader = csv.DictReader(f)
        for name in reader.fieldnames:
            d[name] = []
        for row in reader:
            for name in row:
                if name in d:
                    d[name].append(row[name])
    n = len(d['source_id'])
    if verbose:
        print(u'input file {} has been loaded.'.format(input_file))
        print(u'input file {} contains {} rows.'.format(input_file, n))
    a = np.zeros((n,), dtype=gdr2_csv_dtype)
    for name in a.dtype.fields:
        if a[name].dtype.kind == 'b':
            a[name] = np.bool(d[name] == 'true')
        elif a[name].dtype.kind == 'f':
            try:
                a[name] = d[name]
            except ValueError as e:
                if debug:
                    print(u'irregular input detected: {:>8}, {}'.format(str(a[name].dtype), name))
                for i in range(n):
                    val = d[name][i].strip().lower()
                    if len(d[name][i]) == 0 or val == 'null':
                        a[name][i] = np.nan
                    else:
                        try:
                            a[name][i] = d[name][i]
                        except ValueError as e:
                            print(u'illegal input detected: {:>32} = ({:>8}) {}'.format(name, str(a[name].dtype), d[name][i]))
                            raise e
        elif a[name].dtype.kind == 'i':
            try:
                a[name] = d[name]
            except ValueError as e:
                if debug:
                    print(u'irregular input detected: {:>8}, {}'.format(str(a[name].dtype), name))
                for i in range(n):
                    val = d[name][i].strip().lower()
                    if len(d[name][i]) == 0 or val == 'null':
                        a[name][i] = -1
                    else:
                        try:
                            a[name][i] = d[name][i]
                        except ValueError as e:
                            print(u'illegal input detected: {:>32} = ({:>8}) {}'.format(name, str(a[name].dtype), d[name][i]))
                            raise e
        elif a[name].dtype.kind == 'U':
            a[name] = d[name]
    toc=time.time()
    if verbose:
        print(u'Time cost: {} seconds'.format(toc-tic))
    return a

def load_gdr3_csv(input_file, verbose=False, debug=False):
    tic=time.time()
    d = {}
    with gzip.open(input_file, 'rt') as f:
        # Skip comment lines (YAML header present in GDR3)
        non_comment_lines = [line for line in f if not line.startswith('#')]
    data = StringIO(''.join(non_comment_lines))
    reader = csv.DictReader(data)
    for name in reader.fieldnames:
        d[name] = []
    for row in reader:
        for name in row:
            if name in d:
                d[name].append(row[name])
    n = len(d['source_id'])
    if verbose:
        print(u'input file {} has been loaded.'.format(input_file))
        print(u'input file {} contains {} rows.'.format(input_file, n))
    a = np.zeros((n,), dtype=gdr3_csv_dtype)
    for name in a.dtype.fields:
        if a[name].dtype.kind == 'b':
            a[name] = np.bool(d[name] == 'true')
        elif a[name].dtype.kind == 'f':
            try:
                a[name] = d[name]
            except ValueError as e:
                if debug:
                    print(u'irregular input detected: {:>8}, {}'.format(str(a[name].dtype), name))
                for i in range(n):
                    val = d[name][i].strip().lower()
                    if len(d[name][i]) == 0 or val == 'null':
                        a[name][i] = np.nan
                    else:
                        try:
                            a[name][i] = d[name][i]
                        except ValueError as e:
                            print(u'illegal input detected: {:>32} = ({:>8}) {}'.format(name, str(a[name].dtype), d[name][i]))
                            raise e
        elif a[name].dtype.kind == 'i':
            try:
                a[name] = d[name]
            except ValueError as e:
                if debug:
                    print(u'irregular input detected: {:>8}, {}'.format(str(a[name].dtype), name))
                for i in range(n):
                    val = d[name][i].strip().lower()
                    if len(d[name][i]) == 0 or val == 'null':
                        a[name][i] = -1
                    else:
                        try:
                            a[name][i] = d[name][i]
                        except ValueError as e:
                            print(u'illegal input detected: {:>32} = ({:>8}) {}'.format(name, str(a[name].dtype), d[name][i]))
                            raise e
        elif a[name].dtype.kind == 'U':
            a[name] = d[name]
    toc=time.time()
    if verbose:
        print(u'Time cost: {} seconds'.format(toc-tic))
    return a

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'vd3')
    verbose = False
    debug = False
    dr3 = False
    for opt, val in opts:
        if opt in ['-v']:
            verbose = True
        elif opt in ['-d']:
            debug = True
        elif opt in ['-3']:
            dr3 = True
    input_file = args[0]
    output_file = args[1]
    assert path.exists(input_file)
    assert input_file.lower().endswith('.csv.gz')
    if dr3:
        src = load_gdr3_csv(input_file, verbose, debug)
        out_dtype = gdr3_csv_dtype
    else:
        src = load_gdr2_csv(input_file, verbose, debug)
        out_dtype = gdr2_csv_dtype
    a = np.memmap(output_file, dtype=out_dtype, mode='w+', shape=src.shape)
    a[:] = src[:]
