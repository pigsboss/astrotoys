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

def load_gaia_csv_to_memmap(input_file, output_file, dtype,
                            verbose=False, debug=False, skip_comments=True,
                            chunk_size=100000):
    """流式解析 Gaia CSV.gz → memmap .npy，极限降低内存占用。"""
    tic = time.time()

    # ---------- 第一轮：统计有效行数 ----------
    with gzip.open(input_file, 'rt') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if skip_comments and row and row[0].startswith('#'):
                continue
            header = row
            break
        if header is None:
            raise RuntimeError('CSV 中没有找到标题行')
        n_rows = 0
        for row in reader:
            if skip_comments and row and row[0].startswith('#'):
                continue
            if any(cell.strip() for cell in row):
                n_rows += 1

    if verbose:
        print(f'统计到 {n_rows} 行数据')

    # ---------- 第二层：预分配 memmap ----------
    out = np.memmap(output_file, dtype=dtype, mode='w+', shape=(n_rows,))

    # ---------- 第三层：分块解析并写入 ----------
    with gzip.open(input_file, 'rt') as f:
        reader = csv.reader(f)
        # 再找标题行
        header = None
        for row in reader:
            if skip_comments and row and row[0].startswith('#'):
                continue
            header = row
            break
        col_map = {name: i for i, name in enumerate(header)}

        row_offset = 0
        chunk = []
        for row in reader:
            if skip_comments and row and row[0].startswith('#'):
                continue
            if not any(cell.strip() for cell in row):
                continue

            vals = []
            for name in dtype.names:
                idx = col_map[name]
                cell = row[idx].strip()
                dt = dtype.fields[name][0]

                if dt.kind == 'b':
                    vals.append(cell.lower() == 'true')
                elif dt.kind == 'f':
                    try:
                        vals.append(float(cell) if cell and cell.lower() != 'null' else np.nan)
                    except ValueError:
                        vals.append(np.nan)
                elif dt.kind == 'i':
                    try:
                        vals.append(int(cell) if cell and cell.lower() != 'null' else -1)
                    except ValueError:
                        vals.append(-1)
                elif dt.kind == 'U':
                    vals.append(cell)   # NumPy 自动截断/补齐
                else:
                    vals.append(cell)
            chunk.append(tuple(vals))

            if len(chunk) >= chunk_size:
                chunk_arr = np.array(chunk, dtype=dtype)
                n = chunk_arr.shape[0]
                out[row_offset:row_offset + n] = chunk_arr
                row_offset += n
                chunk.clear()

        # 写入最后剩余行
        if chunk:
            chunk_arr = np.array(chunk, dtype=dtype)
            out[row_offset:row_offset + chunk_arr.shape[0]] = chunk_arr

    out.flush()
    del out

    toc = time.time()
    if verbose:
        print(f'转换完成：{output_file}，耗时 {toc - tic:.1f} 秒')
    return np.memmap(output_file, dtype=dtype, mode='r')

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

    dtype = gdr3_csv_dtype if dr3 else gdr2_csv_dtype
    load_gaia_csv_to_memmap(input_file, output_file, dtype,
                            verbose=verbose, debug=debug,
                            skip_comments=dr3)
