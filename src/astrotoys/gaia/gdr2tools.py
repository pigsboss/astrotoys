#!/usr/bin/env python
#coding=utf-8
"""Gaia Data Release 2 tools.

Syntax: gdr2tools.py [actions] [options]

Actions:
  help     print this help message.
  find     find source files.
  map      make sky maps.
  iometer  random and sequential read throughput benchmark.

Options:
  -s, --source        source directory
  -d, --destination   destination directory
  -O, --overwrite     overwrite existing files
  -p, --pixelization  pixelization scheme (available: p3g, healpix)
  -c, --coordinate    coordinate system (available: equatorial, ecliptical, and galactic)
  -n, --resolution    pixel resolution for both p3g and healpix pixelization
                      for p3g scheme
  -q, --quaternion    attitude quaternion for p3g pixelization
      --span          span angle for p3g pixelization
      --angles        attitude Euler angles for p3g pixelization
      --axis          pointing vector for p3g pixelization
      --up            position vector for p3g pixelization
  -r, --response      pixel response (available: humanvision)
      --nest          healpix pixelization indexing scheme (available: 0, 1)

Copyright: pigsboss@github
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import tables
import sys
import os
import gzip
import colorsys
import astrotoys.observer.humanvision as hv
from astrotoys.photometry import gmag2photrate
from time import time
from multiprocessing import Pool, Queue, Process, cpu_count
from os import path
from subprocess import run, PIPE
from getopt import gnu_getopt
from astropy.io import fits
from pymath.quaternion import fit_attitude, conjugate, ptr2xyz, rotate
from astrotoys.p3g.pixelgrid import SphericalRectangle, PixelGrid
from astrotoys.p3g.skymap import SkyMap, SkyMapList, parse_angle
from astrotoys.formats import gdr2_csv_dtype, gdr2_source_meta_dtype
from astrotoys.wcstime import convert_csys
from astrotoys.gaia.csv2npy import load_gdr2_csv

HEALPIX_MAX_LEVEL = 12
HEALPIX_NEST      = True
DEFAULT_TEFF_VAL  = 5000.0
HEALPIX_COORD     = {'ec':'E', 'eq':'C', 'ga':'G'}


class P3GLocator:
    def __init__(self, quat, span, N, csys):
        self.quat = quat
        try:
            self.span_v, self.span_u = span
        except TypeError:
            self.span_v = self.span_u = span
        try:
            self.NV, self.NU = N
        except TypeError:
            self.NV = self.NU = N
        self.span = (self.span_v, self.span_u)
        self.N    = (self.NV    , self.NU    )
        av, au = 2.0 * np.tan(np.double(self.span) / 2.0)
        self.grid_sz_v, self.grid_sz_u = av, au
        self.px_sz_v, self.px_sz_u = av/self.NV, au/self.NU
        self.csys = csys
        self.shape = (self.NV, self.NU)
        self.size  = self.NV * self.NU

    def __call__(self, ra, dec):
        phi, theta = convert_csys(ra, dec, from_csys='eq', to_csys=self.csys)
        p = np.double(ptr2xyz(ra, dec))
        pref = rotate(conjugate(self.quat), p)
        p_inside = (np.abs(pref[1]) <= np.abs(self.grid_sz_u*0.5*pref[0])) & (np.abs(pref[2]) <= np.abs(self.grid_sz_v*0.5*pref[0])) & (pref[0]>0)
        if np.isscalar(p[0]):
            if p_inside:
                xidx = pref[1] / pref[0] / self.px_sz_u + (self.NU - 1.0)*0.5
                yidx = pref[2] / pref[0] / self.px_sz_v + (self.NV - 1.0)*0.5
            else:
                xidx = np.nan
                yidx = np.nan
        else:
            xidx = np.zeros(p[0].shape)
            yidx = np.zeros(p[0].shape)
            xidx[~p_inside] = np.nan
            yidx[~p_inside] = np.nan
            xidx[p_inside] = pref[1][p_inside] / pref[0][p_inside] / self.px_sz_u + (self.NU - 1.0)*0.5
            yidx[p_inside] = pref[2][p_inside] / pref[0][p_inside] / self.px_sz_v + (self.NV - 1.0)*0.5
        pidx = np.int64(xidx[p_inside]) + np.int64(yidx[p_inside])*self.NU
        return pidx, p_inside

class HPXLocator:
    def __init__(self, nside, nest, csys):
        self.nside = nside
        self.nest  = nest
        self.csys  = csys
        self.size  = hp.nside2npix(nside)
        self.shape = (self.size, )

    def __call__(self, ra, dec):
        phi, theta = convert_csys(ra, dec, from_csys='eq', to_csys=self.csys)
        return hp.ang2pix(self.nside, np.pi/2.0-theta, phi, nest=self.nest), np.ones(np.shape(phi), dtype='bool')

class HumanVision:
    def __init__(self, method='cubic', CMF='10deg'):
        self.observer = hv.CIEObserver(method=method, CMF=CMF)
        self.pixfmt   = np.dtype([('r', 'f8'), ('g', 'f8'), ('b', 'f8')])
    def __call__(self, gdr2src):
        coef = self.observer.K_to_rgb(np.nan_to_num(gdr2src['teff_val'], nan=DEFAULT_TEFF_VAL))
        pixv = np.empty(gdr2src.size, dtype=hv.pixfmt)
        pixv['r'] = gdr2src['phot_g_mean_flux']*coef[0]
        pixv['g'] = gdr2src['phot_g_mean_flux']*coef[1]
        pixv['b'] = gdr2src['phot_g_mean_flux']*coef[2]
        return pixv

class BandpassFlux:
    def __init__(self, band='g'):
        if band.lower()=='g':
            self.bandfield='phot_g_mean_mag'
        elif band.lower()=='r':
            self.bandfield='phot_rp_mean_mag'
        elif band.lower()=='b':
            self.bandfield='phot_bp_mean_mag'
        else:
            raise TypeError('unsupported band {}.'.format(band))
        self.pixfmt = np.dtype([('flux', 'f8')])
    def __call__(self, gdr2src):
        return gmag2photrate(gdr2src[self.bandfield]).astype(self.pixfmt)

def __sources_inside__(meta, level, rect):
    """Check if source file contains GDR2 sources inside specified spherical rectangle.

meta  meta info of source file
level healpix pixelization level (nside=2**level)
rect  pre-defined spherical rectangle object
"""
    nside = 2**level
    return np.any(rect.inside(np.array(hp.pix2vec(
        nside,
        np.int64(range(
            source_id_to_pixel(meta['source_id_start'], level=level),
            source_id_to_pixel(meta['source_id_end'  ], level=level) + 1)),
        nest=HEALPIX_NEST))))

def __count_sources__(qin, qstat, qout, pixel_locator, pixel_response):
    """Map GDR2 source to counts on specific pixel grid.

qin            input queue
qstart         status report queue
qout           output queue
pixel_locator  callable : ra, dec -> pixel index
pixel_response callable : gdr2 source -> pixel response, e.g., BB temp and magnitude to RGB as response of human vision
"""
    srcfile = qin.get()
    cdata   = np.zeros(pixel_locator.shape, dtype=pixel_response.pixfmt)
    while srcfile is not None:
        src = np.memmap(srcfile, dtype=gdr2_csv_dtype, mode='r')
        pidx, inside = pixel_locator(np.deg2rad(src['ra']), np.deg2rad(src['dec']))
        pixv = pixel_response(src[inside])
        if pixv.dtype.fields is None:
            cdata[:] += np.reshape(np.bincount(pidx, pixv, pixel_locator.size), pixel_locator.shape)
        else:
            for field in pixv.dtype.fields:
                cdata[field][:] += np.reshape(np.bincount(pidx, pixv[field], pixel_locator.size), pixel_locator.shape)
        qstat.put(srcfile)
        srcfile = qin.get()
    qout.put(cdata)

def __queue_put__(l, q):
    for a in l:
        q.put(a)

def source_id_to_pixel(source_id, level=HEALPIX_MAX_LEVEL):
    """Compute HEALPix pixel index (NESTED, Celestial coordinate) from GDR2 source ID.

NSIDE = 2**level
NPIX  = 12*NSIDE*NSIDE
"""
    return source_id // (2**35 * 4**(12-level))

def make_spherical_rectangle(quat, span):
    au, av = 2.0*np.tan(np.double(span)/2.0)
    return SphericalRectangle(rotate(quat, [
        [  1.0,   1.0,   1.0,   1.0],
        [-au/2,  au/2,  au/2, -au/2],
        [-av/2, -av/2,  av/2,  av/2]
    ]))

def make_map_p3g(src, quat, span, N, csys, pixel_response, threads):
    pixel_locator = P3GLocator(quat, span, N, csys)
    if path.isdir(src.source):
        cursrc = src.find_source_files_in_spherical_rectangle(quat, span)
        print(u'{} source files found.'.format(cursrc.size))
        q_npy = Queue()
        q_sta = Queue()
        q_map = Queue()
        workers = []
        for i in range(threads):
            proc = Process(target=__count_sources__, args=(q_npy, q_map, pixel_locator, pixel_response))
            proc.start()
            workers.append(proc)
        packs = list(cursrc['filename'])+[None]*threads
        feeder = Process(target=__queue_put__, args=(packs, q_npy))
        feeder.start()
        print('Processing source files......')
        t = 0
        tic = time()
        while t<cursrc.size:
            stat = q_sta.get()
            sys.stdout.write(u'  \r{}/{} ({:.1f}%): {} processed.'.format(t+1, cursrc.size, 100.0*(t+1)/cursrc.size, stat))
            sys.stdout.flush()
            t+=1
        print('\nProcessing source files......Finished ({.1f} seconds).'.format(time()-tic))
        cdata = np.zeros(pixel_locator.shape, dtype=pixel_response.pixfmt)
        print('Collecting results from parallel workers......')
        for i in range(threads):
            if cdata.dtype.fields is None:
                cdata += q_map.get()
            else:
                cdata_new = q_map.get()
                for field in cdata.dtype.fields:
                    cdata[field] += cdata_new[field]
            print('  Results retrieved from thread {}/{}.'.format(i+1, threads))
        print('Collecting results from parallel workers......Finished.')
        for p in workers:
            p.join()
        feeder.join()
        print('All jobs done.')
    else:
        cdata   = np.zeros(pixel_locator.shape, dtype=pixel_response.pixfmt)
        src_in  = src.find_sources_in_spherical_rectangle(quat, span)
        pidx, inside = pixel_locator(np.deg2rad(src_in['ra']), np.deg2rad(src_in['dec']))
        pixv = pixel_response(src_in[inside])
        if pixv.dtype.fields is None:
            cdata[:] = np.reshape(np.bincount(pidx, pixv, pixel_locator.size), pixel_locator.shape)
        else:
            for field in pixv.dtype.fields:
                cdata[field][:] = np.reshape(np.bincount(pidx, pixv[field], pixel_locator.size), pixel_locator.shape)
    return cdata

def make_map_hpx(src, nside, nest, csys, pixel_response, threads):
    q_npy = Queue()
    q_sta = Queue()
    q_map = Queue()
    workers = []
    pixel_locator = HPXLocator(nside, nest, csys)
    for i in range(threads):
        proc = Process(target=__count_sources__, args=(q_npy, q_sta, q_map, pixel_locator, pixel_response))
        proc.start()
        workers.append(proc)
    packs = list(src.content['filename'])+[None]*threads
    feeder = Process(target=__queue_put__, args=(packs, q_npy))
    feeder.start()
    print('Processing source files......')
    t = 0
    tic = time()
    while t<src.content.size:
        stat = q_sta.get()
        sys.stdout.write(u'  \r{}/{} ({:.1f}%): {} processed.'.format(t+1, src.content.size, 100.0*(t+1)/src.content.size, stat))
        sys.stdout.flush()
        t+=1
    print('\nProcessing source files......Finished ({:.1f} seconds).'.format(time()-tic))
    cdata = np.zeros(pixel_locator.shape, dtype=pixel_response.pixfmt)
    print('Collecting results from parallel workers......')
    for i in range(threads):
        if cdata.dtype.fields is None:
            cdata += q_map.get()
        else:
            cdata_new = q_map.get()
            for field in cdata.dtype.fields:
                cdata[field] += cdata_new[field]
        print('  Result retrieved from thread {}/{}.'.format(i+1, threads))
    print('Collecting results from parallel workers......Finished.')
    for p in workers:
        p.join()
    feeder.join()
    print('All jobs done.')
    return cdata

def iometer(cfgfile, destdir, reps=7, seqs=20):
    """Random and sequential throughput benchmark.

cfgfile  configuration file path
destdir  destination directory to save benchmark results
reps     number of repeated runs
seqs     number of sequential read chunks
"""
    ## preparation
    with open(cfgfile, 'r') as cfg:
        sources, labels = list(map(list, zip(*[r.split(';') for r in cfg.read().splitlines()])))
    libraries = []
    for s in sources:
        if s.lower().endswith('csv'):
            libraries.append(GDR2Library(s, suffix='.csv.gz'))
        elif s.lower().endswith('npy'):
            libraries.append(GDR2Library(s, suffix='.npy'))
        else:
            libraries.append(GDR2Library(s))
    ## random throughput
    levels = np.arange(5, 10, dtype='uint64')
    nsides = 2**levels
    npixs  = 12*(nsides**2)
    times  = np.zeros((reps, len(levels), len(labels)))
    iops   = np.zeros_like(times)
    tex = open(path.join(destdir, 'iometer.tex'), 'w')
    csv = open(path.join(destdir, 'iometer_randread.csv'), 'w')
    csv.write("""run,level,label,time (seconds),iops (bytes per second)
""")
    for i in range(reps):
        for j in range(len(levels)):
            res = []
            pix = int(np.random.rand(1) * npixs[j])
            for k in range(len(labels)):
                tic = time()
                src = libraries[k].find_sources_in_healpix(pix, levels[j])
                times[i, j, k] = time()-tic
                res.append(np.sort(src['ra']))
                iops[i, j, k]  = gdr2_csv_dtype.itemsize*src.size / times[i, j, k] ## bytes per second
                print(u'run {}, level {}, {}, {:.2E} seconds (found {} sources)'.format(i+1, levels[j], labels[k], times[i,j,k], src.size))
                csv.write(u'{:d},{:d},{},{:f},{:f}\n'.format(i+1, levels[j], labels[k], times[i,j,k], iops[i,j,k]))
            if len(res)<=5:
                assert np.allclose(*res)
    csv.close()
    databytes = iops*times
    rand_read = np.sum(databytes, axis=0)/np.sum(times, axis=0)*1e-6
    rand_read_err = np.empty_like(rand_read)
    for i in range(len(levels)):
        for j in range(len(labels)):
            rand_read_err[i,j] = (np.sum(1e-6*databytes[:,i,j]*(1e-6*iops[:,i,j] - rand_read[i,j])**2.0) / np.sum(1e-6*databytes[:,i,j])) ** 0.5
    ## sequential throughput
    times = np.zeros((reps, len(labels), seqs))
    iops  = np.zeros_like(times)
    csv = open(path.join(destdir, 'iometer_seqread.csv'), 'w')
    csv.write("""
run,label,time (seconds),iops (bytes per second)
""")
    for i in range(reps):
        for j in range(len(labels)):
            k = 0
            tic = time()
            for a in libraries[j]:
                times[i, j, k] = time()-tic
                iops[i, j, k]  = gdr2_csv_dtype.itemsize*a.size / times[i, j, k]
                csv.write(u'{:d},{},{:f},{:f}\n'.format(i, labels[j], times[i, j, k], iops[i, j, k]))
                tic = time()
                k += 1
                if k>=seqs:
                    break
    csv.close()
    seq_read = np.mean(np.sum(iops*times, axis=2) / np.sum(times, axis=2), axis=0)*1e-6
    seq_read_err = np.std(np.sum(iops*times, axis=2)/np.sum(times, axis=2), axis=0)*1e-6
    tex.write("""\\documentclass{article}
\\begin{document}
\\begin{table}[H]
\\caption{Random read throughput benchmark results}
\\begin{tabular}{r|r|r}
Level & Label & Throughput (Bytes/s) \\\\
\\hline
""")
    for i in range(len(levels)):
        for j in range(len(labels)):
            tex.write(u'{} & {} & {:.2E}$\pm${:.1E} \\\\ \\hline\n'.format(levels[i], labels[j], rand_read[i,j], rand_read_err[i,j]))
    tex.write("""
\\end{tabular}
\\end{table}
\\begin{table}[H]
\\caption{Sequential read throughput benchmark results}
\\begin{tabular}{r|r}
""")
    for i in range(len(labels)):
        tex.write(u'{} & {:.1E}$\pm${:.1E} \\\\ \\hline\n'.format(labels[i], seq_read[i], seq_read_err[i]))
    tex.write("""
\\end{tabular}
\\end{table}
\\end{document}
""")
    tex.close()
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for k in range(len(labels)):
        ax.errorbar(levels, rand_read[:,k], yerr=rand_read_err[:,k], fmt='-o', label=labels[k])
    ax.legend()
    ax.set_title('Random Read Throughput')
    ax.set_xlabel('HEALPix Levels')
    ax.set_ylabel('Throughput, in MBytes/s')
    plt.savefig(path.join(destdir, 'randread.png'))
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.barh(range(len(labels)), seq_read)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Throughput, in MBytes/s')
    ax.set_title('Sequential Read Throughput')
    plt.tight_layout()
    plt.savefig(path.join(destdir, 'seqread.png'))


class GDR2Library:
    """Gaia Data Release 2 source library.

This class is intended to provide a unified interface to access the GDR2 sources.
The following backend repositories are supported.

1. cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv, which is the original repository
   downloaded directly from ESAC (European Space Astronomy Centre).
2. cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/npy, which is local converted NumPy
   structured-array repository. Each .npy file corresponds to a gzipped csv file
   in the original repository, using local defined NumPy data type.
3. a single huge HDF5 file that contains a PyTable object.
"""
    def __init__(self, srcpath, prefix='GaiaSource_', suffix='.npy'):
        if path.isdir(srcpath):
            self.source = path.normpath(path.abspath(path.realpath(srcpath)))
            content = {
                'filename'        : [],
                'filesize'        : [],
                'source_id_start' : [],
                'source_id_end'   : [],
                'sources'         : []
            }
            sys.stdout.write(u'Scanning {}......'.format(self.source))
            sys.stdout.flush()
            result = run([
                'find', self.source, '-type', 'f', '-name', '{}*{}'.format(prefix, suffix), '-ls'
            ], check=True, stdout=PIPE)
            sys.stdout.write(u'\rScanning {}......Finished.\n'.format(self.source))
            sys.stdout.flush()
            sys.stdout.write(u'Parsing filenames......')
            sys.stdout.flush()
            for line in result.stdout.decode().splitlines():
                cols = line.split()
                content['filesize'].append(int(cols[6]))
                content['filename'].append(cols[-1])
                start, end = path.split(cols[-1])[-1][len(prefix):-len(suffix)].split('_')
                content['source_id_start'].append(int(start))
                content['source_id_end'].append(int(end))
                content['sources'].append(int(cols[6]) // gdr2_csv_dtype.itemsize)
            sys.stdout.write(u'\rParsing filenames......Finished.\n')
            sys.stdout.flush()
            self.content=np.zeros(len(content['filename']), dtype=gdr2_source_meta_dtype)
            for field in content:
                self.content[field][:] = content[field]
            self.num_chunks = len(self.content['filename'])
        else:
            srcfile, srctab = srcpath.split(':')
            self.source  = path.normpath(path.abspath(path.realpath(srcfile)))
            self.h5file  = tables.open_file(
                self.source,
                mode='r'
##                chunk_cache_nelments=0,
##                chunk_cache_preempt=0.,
##                chunk_cache_size=0
            )
            self.content = self.h5file.get_node(srctab)
            self.num_chunks = self.content.nrows // self.content.chunkshape[0]

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos < self.num_chunks:
            if path.isdir(self.source):
                fname = self.content['filename'][self.pos]
                if fname.lower().endswith('.npy'):
                    a = np.copy(np.memmap(fname, dtype=gdr2_csv_dtype, mode='r'))
                elif fname.lower().endswith('.csv.gz'):
                    a = np.copy(load_gdr2_csv(fname))
            else:
                a = np.copy(self.content.read(start=self.pos*self.content.chunkshape[0], stop=min((self.pos+1)*self.content.chunkshape[0], self.content.nrows)))
            self.pos += 1
            return a
        else:
            raise StopIteration

    def find_source_files_in_spherical_rectangle(self, quat, span, show_profiling=False):
        """Find source files that may possibly contain certain sources located in specified spherical rectangle.

This function supports directories of .csv.gz files or .npy files as its backend repositories only.

Arguments:
  quat (quaternion) and span (angular size in height x width) define the spherical rectangle.

"""
        assert path.isdir(self.source)
        level = 6
        sv, su = span
        content = np.copy(self.content)
        tic = time()
        with Pool(2*cpu_count()+1) as pool:
            while level<=HEALPIX_MAX_LEVEL:
                nside = 2**level
                rect = make_spherical_rectangle(quat, (sv+4.*hp.nside2resol(nside), su+4.*hp.nside2resol(nside)))
                mask = np.array(list(pool.starmap(__sources_inside__, zip(content, (level,)*content.size, (rect,)*content.size)))) ## multiple threads
                ## mask = np.array(list(map(__sources_inside__, content, (level,)*content.size, (rect,)*content.size))) ## single thread
                content = np.copy(content[mask])
                level += 1
        if show_profiling:
            print(u'found {:d} source files in {:.2f} seconds.'.format(content.size, time()-tic))
        return content

    def find_sources_in_healpix(self, pix, level):
        """Find sources that locate in specified HEALPix pixel.

Arguments:
  pix    is the pixel index of nested HEALPix grid.
  level  is the hierarchical level of the HEALPix grid, (N_side = 2**level, and N_pix = 12 * 2**N_side).

Return:
  a structured ND-array in astrotoys.formats.gdr2_csv_dtype.
"""
        sid_start = int( pix    * (2**35) * (4**(12-level)))
        sid_stop  = int((pix+1) * (2**35) * (4**(12-level)))
        if path.isdir(self.source):
            sources  = np.zeros((0,), dtype=gdr2_csv_dtype)
            for i in range(len(self.content['filename'])):
                if (
                        ((self.content['source_id_start'][i] >= sid_start) and (self.content['source_id_start'][i] <= sid_stop)) or
                        ((self.content['source_id_end'][i]   >= sid_start) and (self.content['source_id_end'][i]   <= sid_stop)) or
                        ((self.content['source_id_start'][i] <= sid_start) and (self.content['source_id_end'][i]   >= sid_stop))
                ):
                    if self.content['filename'][i].lower().endswith('.npy'):
                        a = np.memmap(self.content['filename'][i], dtype=gdr2_csv_dtype, mode='r')
                    elif self.content['filename'][i].lower().endswith('.csv.gz'):
                        a = load_gdr2_csv(self.content['filename'][i])
                    sources = np.concatenate([
                        sources,
                        a[np.logical_and(
                            np.bool_(a['source_id'][:]>=sid_start),
                            np.bool_(a['source_id'][:]< sid_stop))]])
        else:
            sources = self.content.read_where('(source_id>={:d}) & (source_id<{:d})'.format(sid_start, sid_stop))            
        return sources

    def find_sources_in_spherical_rectangle(self, quat, span, show_profiling=True, show_progress=True):
        """Find sources that locate in specified spherical rectangle.

Arguments:
  quat (quaternion) and span (angular size in height x width) define the spherical rectangle.
"""
        g = PixelGrid(quat=quat, span=span, N=4)
        level = int(np.floor(np.log2((4.*np.pi/(4.*np.max(g.px_area))/12.)**.5)))
        nside = 1<<level
        x, y, z = np.concatenate((g.pixel_vertices()), axis=-1)
        pixs  = np.unique(np.ravel(hp.vec2pix(nside, x, y, z, nest=HEALPIX_NEST)))
        if path.isdir(self.source):
            sources = np.zeros((0, ), dtype=gdr2_csv_dtype)
        else:
            sources = np.zeros((0, ), dtype=self.content.dtype)
        tic = time()
        for i in range(pixs.size):
            if show_progress:
                print('processing {:d}/{:d} HEALPIX pixel...'.format(i+1, pixs.size))
            sources = np.concatenate((sources, self.find_sources_in_healpix(pixs[i], level)))
        if show_profiling:
            print(u'found {:d} sources in {:.2f} seconds.'.format(sources.size, time()-tic))
        pxyz = ptr2xyz(np.deg2rad(sources['ra']), np.deg2rad(sources['dec']), 1.)
        pinside = g.boundary.inside(pxyz)
        return sources[pinside]

def main():
    opts, args = gnu_getopt(
        sys.argv[1:],
        's:d:Op:c:n:q:r:t:',
        [
            'source=',
            'destination=',
            'overwrite',
            'pixelization=',
            'coordinate=',
            'resolution=',
            'quaternion=',
            'span=',
            'angles=',
            'axis=',
            'up=',
            'response=',
            'nest=',
            'threads='
        ])
    quat         = None
    phi          = None
    theta        = None
    psi          = None
    axis         = None
    up           = None
    outputfile   = None
    overwrite    = False
    response     = 'humanvision'
    nest         = True
    csys         = 'ecliptical'
    threads      = cpu_count()
    for opt, val in opts:
        if opt in ['-s', '--source']:
            source = val
        elif opt in ['-d', '--destination']:
            destdir = path.normpath(path.abspath(path.realpath(val)))
            if not path.isdir(destdir):
                os.makedirs(destdir)
        elif opt in ['-O', '--overwrite']:
            overwrite = True    
        elif opt in ['-p', '--pixelization']:
            pixelization = val
        elif opt in ['-c', '--coordinate']:
            csys = val
        elif opt in ['-n', '--resolution']:
            try:
                N = int(val)
            except ValueError:
                N = tuple(map(int, val.split(',')))
        elif opt in ['-q', '--quaternion']:
            quat = np.double(list(map(float, val.split(','))))
        elif opt in ['--span']:
            try:
                span = (parse_angle(val),)*2
            except ValueError:
                span = tuple(map(parse_angle, val.split(',')))
        elif opt in ['--angles']:
            phi, theta, psi = map(parse_angle, val.split(','))
        elif opt in ['--axis']:
            axis = map(float, val.split(','))
        elif opt in ['--up']:
            up = map(float, val.split(','))
        elif opt in ['-r', '--response']:
            response = val
        elif opt in ['--nest']:
            nest = bool(val)
        elif opt in ['--overwrite']:
            overwrite = True
        elif opt in ['-t', '--threads']:
            threads = int(val)
    action = args[0]
    if action == 'help':
        print(__doc__)
        sys.exit()
    elif action == 'map':
        src = GDR2Library(source)
        try:
            resolstr = '{:d}x{:d}'.format(N[0], N[1])
        except TypeError:
            resolstr = 'N{:d}'.format(N)
        if response == 'humanvision':
            pixel_response = HumanVision()
        elif response.lower().startswith('flux'):
            _, band = response.lower().split(':')
            pixel_response = BandpassFlux(band)
        if pixelization.lower() == 'p3g':
            quat, axis, up, phi, theta, psi, _ = fit_attitude(quat, axis, up, phi, theta, psi)
            cdata = make_map_p3g(src, quat, span, N, csys, pixel_response, threads)
            print('Create FITS canvas for P3G sky map......')
            sm = SkyMap(quat=quat, axis=axis, up=up, phi=phi, theta=theta, psi=psi, N=N, span=span)
            sm.pprint()
            if cdata.dtype.fields is None:
                sm.cdata[:] = cdata[:]
                outputfile = path.join(destdir, 'P3G_{}.fits'.format(resolstr))
                while path.isfile(outputfile) and (not overwrite):
                    outputfile = input(u'{} exists. Please type in a new path: '.format(outputfile))
                sm.save(outputfile, overwrite=overwrite)
            else:
                for field in cdata.dtype.fields:
                    sm.cdata[:] = cdata[field][:]
                    outputfile = path.join(destdir, 'P3G_{}_{}.fits'.format(resolstr, field.upper()))
                    while path.isfile(outputfile) and (not overwrite):
                        outputfile = input(u'{} exists. Please type in a new path: '.format(outputfile))
                    sm.save(outputfile, overwrite=overwrite)
        elif pixelization.lower() in ['healpix', 'hpx']:
            cdata = make_map_hpx(src, N, nest, csys, pixel_response, threads)
            if cdata.dtype.fields is None:
                outputfile = path.join(destdir, 'HPX_{}.fits'.format(resolstr))
                while path.isfile(outputfile) and (not overwrite):
                    outputfile = input(u'{} exists. Please type in a new path: '.format(outputfile))
                hp.write_map(outputfile, cdata, nest=nest, dtype=np.float64, coord=HEALPIX_COORD[csys[:2]], overwrite=overwrite)
            else:
                for field in cdata.dtype.fields:
                    outputfile = path.join(destdir, 'HPX_{}_{}.fits'.format(resolstr, field.upper()))
                    while path.isfile(outputfile) and (not overwrite):
                        outputfile = input(u'{} exists. Please type in a new path: '.format(outputfile))
                    hp.write_map(outputfile, cdata[field], nest=nest, dtype=np.float64, coord=HEALPIX_COORD[csys[:2]], overwrite=overwrite)
        if response == 'humanvision':
            outputfile = path.join(destdir, '{}_{}_{}.cdata.npy'.format(pixelization, response, resolstr))
            while path.isfile(outputfile) and (not overwrite):
                outputfile = input(u'{} exists. Please type in a new path: '.format(outputfile))
            a = np.memmap(outputfile, shape=cdata.shape, dtype=cdata.dtype, mode='w+')
            a[:] = cdata[:]
            print(u'NumPy ND-array saved to {} for postprocessing.'.format(outputfile))
    elif action == 'iometer':
        cfgfile = args[1]
        iometer(cfgfile, destdir)

if __name__ == '__main__':
    main()
