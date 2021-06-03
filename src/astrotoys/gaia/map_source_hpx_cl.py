#!/usr/bin/env python
#coding=utf-8
"""Make HEALPix sky map from input source (contained in a PyTables HDF5 file) with OpenCL.

Notes
1. Subject: Float-point to integer conversion
   Currently atomic operations are implemented for integers only. Float to integer conversion
   thus is necessary to compute histogram (bin-count) with OpenCL. This is a common task in 
   digital audio processing, where both floating-point samples and integer samples exist.
   A natural way is scaling expected range of floating-point samples to [0, 2**32) (unsigned
   32-bit integer), [0, 2**64) (unsigned 64-bit integer) etc. So the question is, what is the
   expected range of floating-point samples?
   For the phot_g_mean_flux field of GDR2 catalogue, the range is about (8.0, 3.9E9). To estimate
   the expected range of each pixel we should also take the resolution of the pixel grid into
   account. For pixel grid with angular resolution higher than (smaller pixels) 1 degree, the
   expected range of pixel values is about (0.0, 6.4E9).
   As a result, i = int(0.5 * x + 0.5) maps floating-point number x to unsigned 32-bit integer i,
   for x in (0.0, 6.4E9), which is the expected range of pixel values of sky map with practical
   resolution.
   The sum of phot_g_mean_flux of all GDR2 sources is about 9.96E9, which is less than 2**44.
   As a result, i = long(1e6 * x + 0.5) maps floating-point number x to unsigned 64-bit integer i
   and allows at least 1e-6 precision for even the faintest source of the catalogue.
"""
import sys
import math
import tables
import pyopencl as cl
import numpy as np
import healpy as hp
import csv
import matplotlib.pyplot as plt
import psutil
from getopt import gnu_getopt
from os import path
from multiprocessing import cpu_count
from time import time

gdr2_phot_g_mean_flux_min =  8.036644778738776
gdr2_phot_g_mean_flux_max = 3911039555.3979454
magic_number = 7

def find_value(L, x):
    """Search sorted (in ascending order) list L[0,1,...n-1] to find the position of given value x.

if x < L[0], return (-inf, 0);
if x > L[n-1], return (n-1, inf);
if L[0] <= x <= L[-1], find i so that L[i] == x (return (i,)) or L[i] < x < L[i+1] (return (i,i+1)).
"""
    n = len(L)
    if x < L[0]:
        return (-float('inf'), 0)
    if x > L[-1]:
        return (n-1, float('inf'))
    l = 0
    r = n-1
    while (r-l)>1:
        m = int((l+r)/2)
        if x<L[m]:
            r = m
        elif x==L[m]:
            return (m,)
        else:
            l = m
    if x==L[l]:
        return (l,)
    elif x==L[r]:
        return (r,)
    else:
        return (l,r)
            
def host_mem_size():
    """Get physical memory size of shared memory multiprocessor host.
"""
    return psutil.virtual_memory().total

def select_compute_device(device):
    """Select compute device for common platforms.

If device is a pyopencl.Device, return device.
If device is a string, list all available devices and return the first device whose name matches the string.
If device is None, return a preferred device according to the following rules:
  the Accelerator (incl. GPU) device with the most compute units, if multiple accelerators are available;
  the Accelerator (incl. GPU) device, if only one accelerator is available;
  the CPU device.
"""
    if not isinstance(device, cl.Device):
        devs = []
        gpus = []
        for platform in cl.get_platforms():
            for d in platform.get_devices():
                if d.type == cl.device_type.CPU:
                    devs.append(d)
                else:
                    devs.append(d)
                    gpus.append(d)
        if device is None:
            if len(gpus)>1:
                cus = [gpu.max_compute_units for gpu in gpus]
                device = gpus[np.argmax(cus)]
            elif len(gpus)==1:
                device = gpus[0]
            else:
                device = devs[0]
        else:
            for d in devs:
                if device.lower() in d.name.lower():
                    device = d
                    break
    assert isinstance(device, cl.Device)
    return device

class HPXMapper:
    """HEALPix mapper.

input_source   is the path of the input PyTables object, specified in path_to_the_file:/path_to_the_node.
    The input table must be sorted by source_id. A completely sorted index (CSI) of source_id is also required.

output_map     is the output ND-array. A new in-memory NumPy ND-array will be created if output_map is None.

bits           is the number of bits for astrometric (ra, dec) and photometric (e.g., phot_g_mean_flux) parameters.
    Available options:
    32  float32 (single precision floating-point) for ra, dec and phot_g_mean_flux, 
         uint32 (unsigned 32bit integer) for atomic operations (e.g., atomic_add, atomic_inc).
    64  float64 (double precision floating-point) for ra, dec and phot_g_mean_flux,
         uint64 (unsigned 64bit long integer) for atomic operations.

max_mem_pct    is maximum system memory percentage the current object can occupy.

phot_band      is the identifier of selected photometric pass band.
    Available identifiers:
    g   G band (Green), phot_g_mean_flux, available for all GDR2 sources.
    bp  BP band (Blue), phot_bp_mean_flux, available for about 92% GDR2 sources.
    rp  RP band (Red), phot_rp_mean_flux, available for about 92% GDR2 sources.
    (See https://www.cosmos.esa.int/web/gaia/iow_20180316 for above Gaia bands definition.)

fill_missing   fill missing photometric data or not.
    If fill_missing = True, phot_g_mean_flux is used to fill missing data of phot_bp_mean_flux or/and
    phot_rp_mean_flux.

nside          is N_side of all sky HEALPix grid.

granularity    is level of all sky division.
    All sky is divided into a series of regions if granularity >= 0.
    Each region also makes a HEALPix pixel but with lower resolution.
    For example when all sky is divided into 12 regions, each region is also a level-0 pixel.
    Number of regions = 12 * (4 ** granularity).

region_index   is the index of the region where the current mapper instance is assigned.

compute_device is the OpenCL compute device resource assigned to the current mapper instance.
"""
    def __init__(
            self,
            input_source,
            output_map=None,
            bits=64,
            max_mem_pct=25,
            phot_band='g',
            fill_missing=True,
            nside=4096,
            granularity=-1,
            region_index=0,
            compute_device=None,
            profiling=None,
            verbose=True
    ):
        ## set compute device
        self.compute_device = select_compute_device(compute_device)
        ## set data shape and data type
        self.global_nside = nside
        self.global_level = int(np.log2(nside)+0.5)
        self.global_npix  = hp.nside2npix(nside)
        self.granularity  = granularity
        self.region_index = region_index
        if self.granularity>=0:
            self.number_regions = hp.nside2npix(2**self.granularity)
            self.region_npix = self.global_npix // self.number_regions
            self.global_offset  = self.region_index * self.region_npix
        else:
            self.number_regions = 1
            self.region_npix = self.global_npix
            self.global_offset  = 0
        if bits==32:
            self.itype = np.dtype('uint32')
            self.ftype = np.dtype('float32')
        elif bits==64:
            self.itype = np.dtype('uint64')
            self.ftype = np.dtype('float64')
        else:
            raise TypeError(u'unsupported bits mode.')
        if output_map is None:
            ## allocate empty buffer for current region
            self.map_data = np.zeros((self.region_npix,), dtype=self.itype)
        else:
            if output_map.size == self.global_npix:
                ## use allocated global buffer
                self.map_data = output_map[(self.global_offset):(self.global_offset+self.region_npix)].view()
            elif output_map.size == self.region_npix:
                ## use allocated regional buffer
                self.map_data = output_map.view()
            else:
                raise ValueError(u'output_map size ({:d}) matches neigher global_npix ({:d}) nor region_npix ({:d}).'.format(
                    output_map.size, self.global_npix, self.region_npix
                ))
        if phot_band.lower().startswith('g'):
            self.gdr2_field = 'phot_g_mean_flux'
        elif phot_band.lower().startswith('b'):
            self.gdr2_field = 'phot_bp_mean_flux'
            self.fill_missing = fill_missing
        elif phot_band.lower().startswith('r'):
            self.gdr2_field = 'phot_rp_mean_flux'
            self.fill_missing = fill_missing
        else:
            raise ValueError(u'unsupported photometric passband identifier {}.'.format(phot_band))
        if profiling is not None:
            self.profiling = path.normpath(path.abspath(path.realpath(profiling)))
        else:
            self.profiling = None
        ## prepare input file and object
        input_file, input_table = input_source.split(':')
        self.input_file_path = path.normpath(path.abspath(path.realpath(input_file)))
        assert tables.is_pytables_file(self.input_file_path), '{} is not valid PyTables file.'.format(self.input_file_path)
        self.file_object  = tables.open_file(self.input_file_path, mode='r', max_blosc_threads=int(1+2*cpu_count()))
        self.tab = self.file_object.get_node(input_table)
        ## set maximum buffer size (number of rows) for ra, dec and flux, which will copy from host memory to compute device
        ## global memory.
        self.host_mem_size = host_mem_size()
        self.max_enqueue_rows = int(min(
            (self.compute_device.global_mem_size - self.map_data.nbytes) / 3 / self.map_data.itemsize,
            self.compute_device.max_mem_alloc_size / self.map_data.itemsize,
            self.host_mem_size * (max_mem_pct / 100.) / (self.tab.rowsize + 4*self.map_data.itemsize)))
        ## determine required slice of input table
        level     = self.global_level
        pix_start = self.global_offset
        pix_stop  = self.global_offset + self.region_npix
        if level>12:
            pix_start = int(pix_start * 4**(12-level))
            pix_stop  = int(pix_stop  * 4**(12-level))
            level     = 12
        self.source_id_start = pix_start * 2**35 * 4**(12-level)
        self.source_id_stop  = pix_stop  * 2**35 * 4**(12-level)
        self.tab_row_start   = max(min(find_value(self.tab.cols.source_id, self.source_id_start)), 0)
        self.tab_row_stop    = min(max(find_value(self.tab.cols.source_id, self.source_id_stop)),  self.tab.nrows)
        self.enqueue_rows    = 2**int(np.log2(self.max_enqueue_rows))
        nqueues = math.ceil((self.tab_row_stop - self.tab_row_start) / self.enqueue_rows)
        if verbose:
            print(u'{:=^80}'.format(' Input Source '))
            print(u'{:<20}: {}'.format('Filename', self.input_file_path))
            print(u'{:<20}: {}'.format('Table', input_table))
            print(u'{:<20}: {:d}'.format('Number of rows:', self.tab.nrows))
            print(u'{:<20}: {:d}'.format('Chunksize (in rows)', self.tab.chunkshape[0]))
            print(u'{:<20}: {}'.format('Field', self.gdr2_field))
            print(u'{:<20}: [{:d}, {:d})'.format('Source IDs range', self.source_id_start, self.source_id_stop))
            print(u'{:<20}: [{:d}, {:d})'.format('Rows range', self.tab_row_start, self.tab_row_stop))
            print(u'{:=^80}'.format(' Pixel Grid '))
            print(u'{:<20}: {:d}'.format('Global N_side', self.global_nside))
            print(u'{:<20}: {:.2f} arcsec'.format('Resolution', hp.nside2resol(self.global_nside, arcmin=True)*60.0))
            print(u'{:<20}: {:d}'.format('Number of regions', self.number_regions))
            print(u'{:<20}: {:d}'.format('Current region', self.region_index))
            print(u'{:<20}: {:d} MiB'.format('Map size', int(self.map_data.nbytes // 1024**2)))
            print(u'{:=^80}'.format(' Compute Device '))
            print(u'{:<20}: {}'.format('OpenCL platform', self.compute_device.platform.name))
            print(u'{:<20}: {}'.format('Device name', self.compute_device.name))
            print(u'{:<20}: {}'.format('OpenCL version', self.compute_device.version))
            print(u'{:<20}: {:d} MiB'.format('Host memory size', int(self.host_mem_size/1024**2)))
            print(u'{:<20}: {:d} MiB'.format('Global memory size', int(self.compute_device.global_mem_size/1024**2)))
            print(u'{:<20}: {:d} MiB'.format('Maximum buffer size', int(self.compute_device.max_mem_alloc_size/1024**2)))
            print(u'{:<20}: {:d} rows'.format('Maximum enqueue size', self.max_enqueue_rows)) # maximum number of rows to enqueue
            print(u'{:<20}: {:d} rows'.format('Aligned enqueue size', self.enqueue_rows))
            print(u'{:<20}: {:d}'.format('Expected queues', nqueues))
            print(u'{:<20}: {:d}'.format('Compute units', self.compute_device.max_compute_units))
        ## set buffersize
        ## buffers allocated in global memory of compute device:
        ##   ra (float32), dec (float32), flux (float32->uint32) and map_data (uint32)
        assert self.map_data.nbytes <= self.compute_device.max_mem_alloc_size, \
            'map size exceeds max allocatable buffer size of selected compute device.'
        ## prepare OpenCL environment
        ncus = self.compute_device.max_compute_units
        if self.compute_device.type == cl.device_type.GPU:
            ws = 64
            self.global_work_size = ncus * magic_number * ws
            if self.global_work_size > self.enqueue_rows:
                self.global_work_size = self.enqueue_rows
            else:
                while (self.global_work_size < self.enqueue_rows) and ((self.enqueue_rows % self.global_work_size) != 0):
                    self.global_work_size += ws
            self.local_work_size = ws
            is_cpu = 0
        else:
            self.global_work_size = ncus
            self.local_work_size = 1
            is_cpu = 1
        assert self.enqueue_rows % self.global_work_size == 0, u'Input buffer size ({:d}) is not properly aligned (n x {:d}).'.format(
            self.enqueue_rows, self.global_work_size)
        self.context = cl.Context([self.compute_device])
        if self.profiling is None:
            self.queue = cl.CommandQueue(self.context)
        else:
            self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        mf = cl.mem_flags
        self.ra_cl    = cl.Buffer(self.context, mf.READ_ONLY, size=self.enqueue_rows*self.ftype.itemsize)
        self.dec_cl   = cl.Buffer(self.context, mf.READ_ONLY, size=self.enqueue_rows*self.ftype.itemsize)
        self.flux_cl  = cl.Buffer(self.context, mf.READ_ONLY, size=self.enqueue_rows*self.ftype.itemsize)
        self.mdata_cl = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=self.map_data)
        with open(path.join(path.split(path.normpath(path.abspath(path.realpath(__file__))))[0], 'map_source_hpx.c'), 'r') as fp:
            self.device_source = fp.read()
        self.device_build_options = '-w -DIS_CPU={:d} -DNITEMS={:d} -DCOUNT={:d} -DNSIDE={:d} -DNPIX={:d} -DHPX_OFFSET={:d}'.format(
            is_cpu, self.enqueue_rows, int(self.enqueue_rows/self.global_work_size), nside, self.region_npix, self.global_offset)
        if bits == 64:
            self.device_build_options += ' -DREQUIRE_64BIT_PRECISION -DFSCALE=1000000.0'
        else:
            self.device_build_options += ' -DFSCALE=0.5f'
        self.program = cl.Program(self.context, self.device_source).build(self.device_build_options)


    def run(self, verbose=True):
        """Perform planned programs on both host and device sides.

Return:
  map_data
"""
        if self.profiling is not None:
            profile = open(self.profiling, 'w')
            profile.write(u'bytes,rows,memory,kernel,timestamp\n')
        self.run_start = time()
        self.mdata_cl
        t = self.tab_row_start
        if verbose:
            sys.stdout.write(u'  Mapping sources......')
            sys.stdout.flush()
        while t<self.tab_row_stop:
            n = min(self.enqueue_rows, self.tab_row_stop-t)
            buf = self.tab[t:t+n]
            evs_mem = []
            if n < self.enqueue_rows:
                evs_mem.append(cl.enqueue_copy(self.queue, self.ra_cl , np.pad(buf['ra'].astype(self.ftype) , (0, self.enqueue_rows-n))))
                evs_mem.append(cl.enqueue_copy(self.queue, self.dec_cl, np.pad(buf['dec'].astype(self.ftype), (0, self.enqueue_rows-n))))
                if (self.gdr2_field in ['phot_bp_mean_flux', 'phot_rp_mean_flux']) and self.fill_missing:
                    flux = np.copy(buf[self.gdr2_field])
                    is_nan = np.isnan(flux)
                    flux[is_nan] = buf['phot_g_mean_flux'][is_nan]
                    evs_mem.append(cl.enqueue_copy(self.queue, self.flux_cl, np.pad(flux.astype(self.ftype), (0, self.enqueue_rows-n))))
                else:
                    evs_mem.append(cl.enqueue_copy(self.queue, self.flux_cl, np.pad(buf[self.gdr2_field].astype(self.ftype), (0, self.enqueue_rows-n))))
            else:
                evs_mem.append(cl.enqueue_copy(self.queue, self.ra_cl,   buf['ra'].astype(self.ftype) ))
                evs_mem.append(cl.enqueue_copy(self.queue, self.dec_cl,  buf['dec'].astype(self.ftype)))
                if (self.gdr2_field in ['phot_bp_mean_flux', 'phot_rp_mean_flux']) and self.fill_missing:
                    flux = np.copy(buf[self.gdr2_field])
                    is_nan = np.isnan(flux)
                    flux[is_nan] = buf['phot_g_mean_flux'][is_nan]
                    evs_mem.append(cl.enqueue_copy(self.queue, self.flux_cl, flux.astype(self.ftype)))
                else:
                    evs_mem.append(cl.enqueue_copy(self.queue, self.flux_cl, buf[self.gdr2_field].astype(self.ftype)))
            ev_exec = self.program.map_source(
                self.queue,
                (self.global_work_size,),
                (self.local_work_size,),
                self.ra_cl,
                self.dec_cl,
                self.flux_cl,
                self.mdata_cl
            )
            t += n
            if verbose:
                sys.stdout.write(u'\r  Mapping sources......{:d}/{:d} rows ({:5.1f}%, {:.2f} MRows/s, {:.2f} GiB/s)'.format(
                    t-self.tab_row_start,
                    self.tab_row_stop-self.tab_row_start,
                    100.0*(t-self.tab_row_start)/(self.tab_row_stop-self.tab_row_start),
                    1e-6*(t-self.tab_row_start)/(time()-self.run_start),
                    1e-9*(t-self.tab_row_start)*self.tab.rowsize/(time()-self.run_start)
                ))
                sys.stdout.flush()
            if self.profiling is not None:
                ev_exec.wait()
                ev_mem_time = 0.
                for ev in evs_mem:
                    ev_mem_time += 1e-9*(ev.profile.end - ev.profile.start)
                profile.write(u'{:d},{:d},{:f},{:f},{:f}\n'.format(
                    n*self.tab.rowsize,
                    n,
                    ev_mem_time,
                    1e-9*(ev_exec.profile.end - ev_exec.profile.start),
                    time()-self.run_start
                ))
        if verbose:
            sys.stdout.write(u'\r  Mapping sources......{:d}/{:d} rows ({:5.1f}%, {:.2f} MRows/s, {:.2f} GiB/s) OK.\n'.format(
                t-self.tab_row_start,
                self.tab_row_stop-self.tab_row_start,
                100.0*(t-self.tab_row_start)/(self.tab_row_stop-self.tab_row_start),
                1e-6*(t-self.tab_row_start)/(time()-self.run_start),
                1e-9*(t-self.tab_row_start)*self.tab.rowsize/(time()-self.run_start)
            ))
            sys.stdout.flush()
        if self.profiling is not None:
            profile.close()
        cl.enqueue_copy(self.queue, self.map_data, self.mdata_cl)
        self.run_end = time()
        return self.map_data

    def profile(self):
        perf = {
            'bytes':[],
            'rows':[],
            'memory':[],
            'kernel':[],
            'timestamp':[]
        }
        with open(self.profiling, 'r') as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                for k in r:
                    perf[k].append(r[k])
        tt    = np.diff(np.pad(np.double(perf['timestamp']), (1,0)))
        gbps  = 1e-9*np.double(perf['bytes']) / tt
        mrows = 1e-6*np.double(perf['rows']) / tt
        mt    = np.double(perf['memory'])
        kt    = np.double(perf['kernel'])
        mbw   = self.enqueue_rows * self.ftype.itemsize * 3 / mt
        sbw   = self.enqueue_rows / kt
        print(u'{:=^50}'.format(' Profiling '))
        print(u'{:<30}: {:.2f} GiB/s'.format('Host table bandwidth', np.mean(gbps[:-1])))
        print(u'{:<30}: {:.2f} MRows/s'.format('Source retrieval rate', np.mean(mrows[:-1])))
        print(u'{:<30}: {:.2f} GiB/s'.format('Device memory bandwidth', 1e-9*np.mean(mbw[:-1])))
        print(u'{:<30}: {:.2f} GRows/s'.format('Kernel mapping rate', 1e-9*np.mean(sbw[:-1])))
        print(u'{:<30}: {:.2f} %'.format('Compute device load', 100.0*np.sum(kt)/np.sum(tt)))

    def __del__(self):
        if self.file_object.isopen:
            self.file_object.close()
