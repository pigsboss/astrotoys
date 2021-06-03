#!/usr/bin/env python
#coding=utf-8
"""Download http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html and prepare static color dictionary.
"""
from subprocess import run
from os import path
import sys
import numpy as np

bbr_color_dtype = np.dtype([
    ('K', 'f4'),
    ('R', 'f4'),
    ('G', 'f4'),
    ('B', 'f4')
])

def main():
    modpath=path.split(path.normpath(path.abspath(path.realpath(__file__))))[0]
    htmlpath = path.join(modpath, 'bbr_color.html')
    if not path.isfile(htmlpath):
        run(['wget', 'http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html',
             '-O', htmlpath
        ], check=True)
    fields = {
        'K'    :2,
        'CMF'  :4,
        'x'    :5,
        'y'    :6,
        'P'    :7,
        'R'    :8,
        'G'    :9,
        'B'    :10,
        'r'    :11,
        'g'    :12,
        'b'    :13,
        '#rgb' :14
    }
    d2  = {}
    d10 = {}
    for f in fields:
        d2[f]  = []
        d10[f] = []
    with open(htmlpath, 'r') as fp:
        for l in fp:
            if l.startswith('<span'):
                cols = l.split()
                if cols[fields['CMF']] == '2deg':
                    for f in fields:
                        d2[f].append(cols[fields[f]])
                elif cols[fields['CMF']] == '10deg':
                    for f in fields:
                        d10[f].append(cols[fields[f]])
    t2 = np.memmap(path.join(modpath, 'bbr_color_2deg.npy'), shape=len(d2['K']), dtype=bbr_color_dtype, mode='w+')
    t2['K'] = np.float32(d2['K'])
    t2['R'] = np.float32(d2['R'])
    t2['G'] = np.float32(d2['G'])
    t2['B'] = np.float32(d2['B'])
    t10 = np.memmap(path.join(modpath, 'bbr_color_10deg.npy'), shape=len(d2['K']), dtype=bbr_color_dtype, mode='w+')
    t10['K'] = np.float32(d10['K'])
    t10['R'] = np.float32(d10['R'])
    t10['G'] = np.float32(d10['G'])
    t10['B'] = np.float32(d10['B'])
    
if __name__ == '__main__':
    main()

