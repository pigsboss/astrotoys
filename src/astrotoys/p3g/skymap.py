#!/usr/bin/env python
#coding=utf-8
"""Planar Projection Pixel Grid (p3g) sky map object and tools.

Syntax: skymap.py [action] [options]

Actions:
  help  print this help message.
  init  create empty sky map or all-sky tessellation sky maps.
  print print meta info of sky map(s).
  show  show 2D image of sky map(s).

Options:
  -c, --csys                coordinate system (Default: equatorial).
  -s, --span                span angle of individual pixel grid.
  -N, --res, --resolution   pixel resolution of individual pixel grid.
  -M, --tes, --tessellation tessellation granularity.
  -i, --in, --input         input FITS file path.
  -o, --out, --ouput        output FITS file path.
  -O                        output FITS file path, overwrite existing file.
  -q, --quat, --quaternion  attitude quaternion of pixel grid.
  --angles                  attitude Euler angles of pixel grid.
  --axis, --up              pointing and position vectors of pixel grid.
  --hdu                     HDU index of input FITS file.
  --allsky                  all-sky tessellation.
  --overwrite               overwrite existing file.

Copyright: pigsboss@github
"""
import healpy as hp
import sys
from getopt import gnu_getopt
from astrotoys.p3g.pixelgrid import *
from astrotoys.wcstime import convert_csys
from pymath.common import triangle_area
from scipy import interpolate
from matplotlib import colors, cm
from datetime import datetime
from astropy.io import fits
from tempfile import mkdtemp
from os import path

DEFAULT_CSYS = 'equatorial'

def mollweide_canvas(npts=NUM_PTS):
    """Generate mollweide canvas
"""
    a = np.pi
    b = np.pi*0.5
    xgv = np.double(range(2*npts))/np.double(2*npts-1)*np.pi*2.0-np.pi
    ygv = np.double(range(  npts))/np.double(  npts-1)*np.pi-np.pi*0.5
    x,y = np.meshgrid(xgv,ygv)
    phi, _, rho = xyz2ptr(x,y,0.0)
    rho_max = a * b / np.sqrt((a*np.sin(phi)) ** 2 + (b*np.cos(phi)) ** 2)
    return x,y,(rho<=rho_max)

class SkyMap(object):
    """A SkyMap instance is a map defined on a pixel grid.
"""
    def __init__(
            self,
            map_data      = np.zeros((4,4)),
            fitshdu       = None,
            fitsfile      = None,
            HDUidx        = 0,
            csys          = DEFAULT_CSYS,
            only_boundary = False,
            **kwargs):
        if fitshdu is not None:
            self.from_fitshdu(fitshdu, only_boundary=only_boundary)
        elif fitsfile is not None:
            self.load(fitsfile, HDUidx=HDUidx, only_boundary=only_boundary)
        else:
            pxg = PixelGrid(**kwargs)
            self.grid  = pxg
            self.__set_map_data__(map_data)
            self.csys  = csys.lower()

    def __set_map_data__(self, map_data):
        """Set map data of the current sky map.
"""
        dv,du = map_data.shape
        v = (np.arange(dv) - np.double(dv-1)*0.5) / np.double(dv)*self.grid.grid_sz_v
        u = (np.arange(du) - np.double(du-1)*0.5) / np.double(du)*self.grid.grid_sz_u
        f = interpolate.RectBivariateSpline(v,u,map_data)
        self.cdata = f(self.grid.v, self.grid.u)

    def __load_events__(self, phi_events, theta_events):
        self.cdata = np.zeros(self.cdata.shape)
        gx,gy,gm = self.grid.locate(phi_events,theta_events)
        gxi,gyi = np.int64(np.round(gx[gm])),np.int64(np.round(gy[gm]))
        for k in range(0,gxi.size):
            self.cdata[gyi[k],gxi[k]] = self.cdata[gyi[k],gxi[k]] + 1.0

    def from_fitshdu(self,hdu,only_boundary=False):
        """Load SkyMap map data and WCS parameters from a FITS HDU.
Map data loaded from the FITS file is assigned to the SkyMap instance
directly. WCS parameters are used to build its pixel grid.
Thus the FITS HDU must meet the following requirements:
  1. NAXIS == 2. Only 2-D images are supported.
  2. NAXIS1 == NAXIS2.
  3. If exists, CTYPE1 must be one of RA---TAN, ELON--TAN, and GLON--TAN.
     CTYPE2 must conform to CTYPE1.
  4. If exist, both CRPIX1 and CRPIX2 must equal to (NAXISi+1)/2.
     The reference point must lie on the center of the image array.
  5. If both exist, CDELT1 and CDELT2 must equal to each other.
If not specified explicitly the primary HDU will be verified and loaded
if it is valid.
"""
        assert hdu.header['NAXIS'] == 2, 'data of HDU must be 2D image array.'
        N  = (hdu.header['NAXIS2'],hdu.header['NAXIS1'])
        NU = hdu.header['NAXIS1']
        NV = hdu.header['NAXIS2']
        span = PIXEL_GRID_SPAN
        phi_0 = 0
        theta_0 = 0
        psi = 0
        if 'CTYPE1' in hdu.header:
            if hdu.header['CTYPE1'].upper()[0:8] == 'ELON-TAN':
                self.csys = 'ecliptic'
            elif hdu.header['CTYPE1'].upper()[0:8] == 'GLON-TAN':
                self.csys = 'galactic'
            elif hdu.header['CTYPE1'].upper()[0:8] == 'RA---TAN':
                self.csys = DEFAULT_CSYS
            else:
                raise ValueError('unsupported CTYPE: {}'.format(hdu.header['CTYPE1']))
        CCPIX1 = (np.double(hdu.header['NAXIS1'])+1.0)*0.5
        CCPIX2 = (np.double(hdu.header['NAXIS2'])+1.0)*0.5
        if 'CDELT1' in hdu.header:
            px_sz_u = np.abs(hdu.header['CDELT1']*np.pi/180.0)
            span_u  = 2.0 * np.arctan(0.5 * np.double(px_sz_u) * np.double(NU))
        if 'CDELT2' in hdu.header:
            px_sz_v = np.abs(hdu.header['CDELT2']*np.pi/180.0)
            span_v  = 2.0 * np.arctan(0.5 * np.double(px_sz_v) * np.double(NV))
        if 'CRVAL1' in hdu.header:
            phi_0 = (hdu.header['CRVAL1']+hdu.header['CDELT1']*(CCPIX1-hdu.header['CRPIX1']))*np.pi/180.0
        if 'CRVAL2' in hdu.header:
            theta_0 = (hdu.header['CRVAL2']+hdu.header['CDELT2']*(CCPIX2-hdu.header['CRPIX2']))*np.pi/180.0
        if 'LONPOLE' in hdu.header:
            psi = (180.0 - hdu.header['LONPOLE'])*np.pi/180.0
        if only_boundary:
            self.grid = PixelGrid(span=(span_v,span_u), N=(4,4),   phi=phi_0, theta=theta_0, psi=psi)
        else:
            self.grid = PixelGrid(span=(span_v,span_u), N=(NV,NU), phi=phi_0, theta=theta_0, psi=psi)
            self.cdata = hdu.data

    def fitshdu(self):
        """Convert current SkyMap instance to an equivalent FITS HDU.
"""
        hdu = fits.ImageHDU(self.cdata)
        hdu.header['BITPIX'] = (-64, 'Double precision floating point')
        hdu.header['BUNIT']  = ('count   ', 'count of events')
        hdu.header['NAXIS']  = (2, '2-dimensional image')
        hdu.header['NAXIS1'] = (self.grid.NU, 'x axis (u axis, or longitudinal axis)')
        hdu.header['NAXIS2'] = (self.grid.NV, 'y axis (v axis, or latitudinal axis)')
        if self.csys.lower().startswith('eq'):
            hdu.header['CTYPE1']  = ('RA---TAN', 'Right ascension, gnomonic projection')
            hdu.header['CTYPE2']  = ('DEC--TAN', 'Declination, gnomonic projection')
            hdu.header['RADESYS'] = ('FK5     ', 'Mean IAU 1984 equatorial coordinates')
        elif self.csys.lower().startswith('ec'):
            hdu.header['CTYPE1']  = ('ELON-TAN', 'Ecliptical longitude, gnomonic projection')
            hdu.header['CTYPE2']  = ('ELAT-TAN', 'Ecliptical latitude, gnomonic projection')
            hdu.header['RADESYS'] = ('FK5     ', 'Mean IAU 1984 ecliptic coordinates')
        elif self.csys.lower().startswith('ga'):
            hdu.header['CTYPE1']  = ('GLON-TAN', 'Galactic longitude, gnomonic projection')
            hdu.header['CTYPE2']  = ('GLAT-TAN', 'Galactic latitude, gnomonic projection')
        else:
            raise ValueError('Unrecognized coordinate systems.')
        hdu.header['CRPIX1'] = ((np.double(self.grid.NU)+1.0)*0.5, 'Pixel coordinate of reference point')
        hdu.header['CRPIX2'] = ((np.double(self.grid.NV)+1.0)*0.5, 'Pixel coordinate of reference point')
        hdu.header['CDELT1'] = ((self.grid.px_sz_u)*180.0/np.pi,   'Pixel width in physical world')
        hdu.header['CDELT2'] = ((self.grid.px_sz_v)*180.0/np.pi,   'Pixel width in physical world')
        hdu.header['CUNIT1'] = ('deg     ',                        'Angles are degrees always')
        hdu.header['CUNIT2'] = ('deg     ',                        'Angles are degrees always')
        phi,theta,_ = xyz2ptr(*tuple(self.grid.axis))
        hdu.header['CRVAL1']  = (phi*180.0/np.pi,                       'Physical coordinate of reference point')
        hdu.header['CRVAL2']  = (theta*180.0/np.pi,                     'Physical coordinate of reference point')
        hdu.header['LONPOLE'] = (180.0-self.grid.psi*180.0/np.pi,       'Native longitude of WCS pole')
        hdu.header['DATE']    = (datetime.isoformat(datetime.utcnow()), 'UTC date of HDU creation')
        return hdu

    def save(self, fitsfile, **kwargs):
        """Save the current SkyMap instance to a new FITS file.
Parameters
  fitsfile is filename of the FITS file.
Optional keywords
  overwrite controls fits.writeto() to either overwrite the FITS file
  or to raise an exception if the FITS file already exists.
  checksum and output_verify:
  refer to fits handbook for details.
"""
        hdulst = fits.HDUList()
        hdulst.append(self.fitshdu())
        hdulst.writeto(fitsfile,**kwargs)

    def load(self,fitsfile,HDUidx=0,only_boundary=False):
        """Load SkyMap map data and WCS parameters from a FITS file.
"""
        hdulst = fits.open(fitsfile,mode='readonly')
        assert HDUidx < len(hdulst), 'HDU index exceeds length of HDU list.'
        self.from_fitshdu(hdulst[HDUidx],only_boundary=only_boundary)
        hdulst.close()

    def exposure_map(self, precession):
        """Get exposure map of the current sky map.
exposure = pixel_area / exposure_area_per_second
exposure_area_per_second = 2 * pi * precession_rate * cos(t),
where t is effective cosine of ecliptical latitude of each pixel.
t is calculated as:
t = (|cos(theta) + a/2| + |cos(theta) - a/2|) / 2,
where a is side length of each pixel.
"""
        _,theta = convert_csys(self.grid.phi, self.grid.theta, from_csys=self.csys, to_csys='ecliptical')
        t = np.abs(np.cos(theta) + self.grid.px_sz_v*0.5)*0.5 + np.abs(np.cos(theta) - self.grid.px_sz_v*0.5)*0.5
        emap = self.grid.px_area / (2.0*np.pi*precession*t)
        return emap

    def exposure_map_projection(self, precession):
        """Get exposure map of the current sky map.
exposure = T * pixel_area_in_cylindrical_map / 4*pi^2,
where T is total time of survey.
TODO: calculate area of pixel containing singular point.
"""
        #
        # get 4 vertices of each pixel:
        v0,v1,v2,v3 = self.grid.pixel_vertices()
        #
        # longitudes and latitudes of vertices in ecliptical coordinate
        # system:
        lon0,lat0 = convert_csys(*tuple(xyz2ptr(*tuple(v0)))[0:2], from_csys=self.csys,to_csys='ecliptic')
        lon1,lat1 = convert_csys(*tuple(xyz2ptr(*tuple(v1)))[0:2], from_csys=self.csys,to_csys='ecliptic')
        lon2,lat2 = convert_csys(*tuple(xyz2ptr(*tuple(v2)))[0:2], from_csys=self.csys,to_csys='ecliptic')
        lon3,lat3 = convert_csys(*tuple(xyz2ptr(*tuple(v3)))[0:2], from_csys=self.csys,to_csys='ecliptic')
        idx = (np.abs(lon0-lon1) > np.pi)
        lon1[idx] = lon1[idx] + np.sign(lon0[idx]) * np.pi * 2.0
        idx = (np.abs(lon0-lon2) > np.pi)
        lon2[idx] = lon2[idx] + np.sign(lon0[idx]) * np.pi * 2.0
        idx = (np.abs(lon0-lon3) > np.pi)
        lon3[idx] = lon3[idx] + np.sign(lon0[idx]) * np.pi * 2.0
        #
        # construct triangles on cylindrical map:
        v0,v1,v2,v3 = np.array([lon0,lat0]),np.array([lon1,lat1]), np.array([lon2,lat2]),np.array([lon3,lat3])
        #
        # calculate pixel areas on cylindrical map:
        ta_d = triangle_area(v0,v1,v2)
        ta_u = triangle_area(v2,v3,v0)
        pxa  = ta_d + ta_u
        #
        # locate singular (polar) point:
        sx,sy,inside = self.grid.locate([0,0],[-np.pi/2.0,np.pi/2.0])
        if inside.any():
            # singular point found:
            sx = np.int64(np.round(sx[inside]))
            sy = np.int64(np.round(sy[inside]))
            pxa[sy,sx] = np.pi*(np.pi - np.abs(lat0[sy,sx]+lat1[sy,sx]+lat2[sy,sx]+lat3[sy,sx])/2.0)
        return pxa / precession / np.pi * 0.5

    def show2D(
            self,
            axes       = None,
            csys       = DEFAULT_CSYS,
            xlabel     = r'$l$',
            ylabel     = r'$b$',
            cmap       = cm.gray,
            local_only = True,
            npts       = None,
            **kwargs):
        """Show data on 2D mollweide map.
=================================
Syntax:
mm,mc,mg = show2D(fig,npts,axes_fontsize,xlabel,ylabel,cmap,...)
=================================
Parameters:
mm is 2D mollweide map.
mc is canvas mask (True: in mollweide canvas).
mg is grid mask (True: in sky map grid).
"""
        if npts is None:
            npts = np.int(np.ceil(np.pi / self.grid.px_sz_v))
        mx,my,mc = mollweide_canvas(npts)
        mm,phi,theta = np.zeros(mc.shape),np.zeros(mc.shape),np.zeros(mc.shape)
        phi[mc], theta[mc] = convert_csys(*tuple(imollweide(mx[mc], my[mc])), from_csys=csys,to_csys=self.csys)
        gx,gy = np.zeros(mc.shape),np.zeros(mc.shape)
        mg = np.zeros((mc.shape),dtype=np.bool)
        gx[~mc],gy[~mc],mg[~mc] = np.nan,np.nan,False
        gx[mc], gy[mc], mg[mc] = self.grid.locate(phi[mc], theta[mc])
        gxi,gyi = np.int64(np.round(gx[mg])),np.int64(np.round(gy[mg]))
        mm[mg] = self.cdata[gyi,gxi]
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        if local_only:
            ma = np.ma.array(mm,mask=~mg)
        else:
            ma = np.ma.array(mm,mask=~mc)
        cmap.set_bad('w', 0.0)
        axes.imshow(np.flipud(ma), extent=[-180.,180.,-90.,90.], cmap=cmap, **kwargs)
        axes.set_xlabel('{}, in deg'.format(xlabel))
        axes.set_ylabel('{}, in deg'.format(ylabel))
        if draw_now:
            plt.show()
        return mm,mc,mg

    def show3D(
            self,
            axes           = None,
            xlabel         = r'$x$',
            ylabel         = r'$y$',
            zlabel         = r'$z$',
            grid_on        = True,
            grid_nlons     = NUM_LON_GRIDS,
            grid_nlats     = NUM_LAT_GRIDS,
            grid_npts      = NUM_PTS,
            grid_linespecs = 'k-',
            grid_linewidth = 0.5,
            cmap           = cm.gray,
            **kwargs):
        """Show data on 3D spherical surface.
"""
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            draw_now = True
        else:
            draw_now = False
        cust_cmap = cm.ScalarMappable(norm=colors.Normalize(vmin=self.cdata.min(),vmax=self.cdata.max()),cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if grid_on:
            plot_globe_grids(
                axes      = axes,
                nlons     = grid_nlons,
                nlats     = grid_nlats,
                npts      = grid_npts,
                linespecs = grid_linespecs,
                linewidth = grid_linewidth)
        ax.plot_surface(self.grid.x, self.grid.y, self.grid.z, facecolors=cust_cmap.to_rgba(self.cdata), **kwargs)
        if draw_now:
            plt.show()
        return axes

    def view(self, csys=DEFAULT_CSYS, projection='planar', npts=None, grid=None):
        """Get a specific view of the current sky map.
A view of a sky map is defined by a celestial coordinate system along
side with a projection from the celestial surface to a 2-D plane.
The data defined on the view is called canvas data.

Supported celestial coordinate system:
Ecliptical, Galactic, and Equatorial.
Supported projection:
Mollweide, Cylindrical and Planar.
============
Syntax:
cdata,x,y,cmask,gmask = SkyMap.view()

Parameters:
csys is coordinate system of the view.
projection is projection of the view.
npts is number of points along shorter axis of the view.
grid is the PixelGrid instance for plane projection.

Return:
cdata is canvas data.
x and y are coordinates of the view.
cmask is logical mask of cdata that indicates if the pixel is defined.
gmask is logical mask of cdata that indicates if the pixel is inside the grid
of the sky map.
"""
        if npts is None:
            npts = np.int(np.ceil(np.pi / self.grid.px_sz_v))
        if grid is None:
            grid = self.grid
        if projection.lower().startswith('p'):
            if grid is self.grid:
                x = self.grid.x
                y = self.grid.y
                cdata = self.cdata
                cmask = np.ones(cdata.shape,dtype=bool)
                gmask = np.ones(cdata.shape,dtype=bool)
            else:
                x,y = np.meshgrid(grid.u,grid.v)
                gx,gy,gmask = self.grid.vec2pix(grid.x,grid.y,grid.z)
                cdata = np.zeros(x.shape)
                gxi = np.int64(np.round(gx[gmask]))
                gyi = np.int64(np.round(gy[gmask]))
                cdata[gmask] = self.cdata[gyi,gxi]
                cmask = np.ones(gmask.shape,dtype=bool)
        else:
            if projection.lower().startswith('c'):
                x,y,cmask = cylindrical_canvas(npts)
                cdata,phi,theta = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)
                phi[cmask], theta[cmask] = convert_csys(x[cmask], y[cmask], from_csys=csys, to_csys=self.csys)
            elif projection.lower().startswith('m'):
                x,y,cmask = mollweide_canvas(npts)
                cdata,phi,theta = np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)
                phi[cmask], theta[cmask] = convert_csys(*tuple(imollweide(x[cmask], y[cmask])), from_csys=csys, to_csys=self.csys)
            else:
                raise ValueError('unrecognized projection: {}'.format(projection))
            gx,gy = np.zeros(x.shape),np.zeros(x.shape)
            gmask = np.zeros((x.shape),dtype=np.bool)
            gx[~cmask],gy[~cmask],gmask[~cmask] = np.nan, np.nan, False
            gx[cmask], gy[cmask], gmask[cmask] = self.grid.locate(phi[cmask], theta[cmask])
            gxi,gyi = np.int64(np.round(gx[gmask])), np.int64(np.round(gy[gmask]))
            cdata[gmask] = self.cdata[gyi,gxi]
        return cdata,x,y,cmask,gmask

    def pprint(self):
        self.grid.pprint()
        print('{:<20}:{:<40}'.format('cdata type', str(self.cdata.dtype)))
        print('{:<20}:{:> 10,.4e}'.format('cdata mean', np.mean(self.cdata)))
        print('{:<20}:{:> 10,.4e}'.format('cdata min', np.min(self.cdata)))
        print('{:<20}:{:> 10,.4e}'.format('cdata max', np.max(self.cdata)))
        print('{:<20}:{:> 10,.4e}'.format('cdata std', np.std(self.cdata)))
        print('{:<20}:{:> 10,.4e}'.format('cdata sum', np.sum(self.cdata)))

    def info(self):
        self.pprint()

class SkyMapList(list):
    """A SkyMapList instance is a iterable container of a series of SkyMap
objects.
To initiate a SkyMapList, either use the following syntax to explicitly
name all its members, i.e., a series of SkyMap objects:
SkyMapList(sm1, sm2, ...)
or provide a set of HEALPix data alongside with the essential parameters
to create define a closed coverage of the celestial sphere.
Keyword-value Parameters:
hpxdata is an numpy array or a list. It's the HEALPix data.
M is the number of SkyMap objects along a meridian every 90 degrees.
N is the number of pixels along each axis of the pixel grid of each SkyMap
objects.
span is the angular span of the pixel grid of each SkyMap object.
nest is a boolean option, which specifies if the pixel indices of input
HEALPix image are nested or not.
"""
    def __init__(
            self,
            *args,
            M       = None,
            N       = NUM_PIX,
            csys    = DEFAULT_CSYS,
            span    = PIXEL_GRID_SPAN,
            hpxdata = None,
            hpxcsys = DEFAULT_CSYS,
            nest    = HEALPIX_NEST):
        if M is None:
            if np.iterable(args) == 1:
                if len(args) > 0:
                    assert isinstance(args[0], SkyMap), '{} is not SkyMap object.'.format(args[0])
                    list.__init__(self, args)
                else:
                    list.__init__(self)
            else:
                list.__init__(self)
        else:
            if hpxdata is None:
                print('Create empty sky map list on user specified all-sky tessellation.')
                phi, theta = tessellangles(M)
                for k in range(len(phi)):
                    self.append(SkyMap(
                        N     = N,
                        span  = span,
                        phi   = phi[k],
                        theta = theta[k],
                        psi   = 0.0,
                        csys  = csys
                    ))
                    print(u'SkyMap {}/{} created.'.format(k+1, len(phi)))
            else:
                self.from_healpix(
                    hpxdata,
                    hpxcsys = hpxcsys,
                    M       = M,
                    N       = N,
                    csys    = csys,
                    span    = span,
                    nest    = nest
                )

    def save(self, fitsfile, **kwargs):
        """Save the current SkyMapList instance to a new FITS file.
Parameters
  fitsfile is filename of the FITS file.
Optional keywords
  overwrite controls fits.writeto() to either overwrite the FITS file
  or to raise an exception if the FITS file already exists.
  checksum and output_verify:
  refer to fits handbook for details.
"""
        hdulst = fits.HDUList()
        for sm in self:
            if isinstance(sm,SkyMap):
                hdulst.append(sm.fitshdu())
        hdulst.writeto(fitsfile,**kwargs)
        print(u'Current SkyMapList instance saved to {}'.format(fitsfile))

    def load(self, fitsfile, HDUidx=None, only_boundary=False):
        """Load SkyMapList from a FITS file.
Loaded SkyMaps are appended to the current SkyMapList.
"""
        hdulst = fits.open(fitsfile, mode='readonly')
        if HDUidx is None:
            HDUidx = range(len(hdulst))
        for k in HDUidx:
            self.append(SkyMap(fitshdu = hdulst[k], only_boundary=only_boundary))
            print(u'{}/{} HDU loaded into SkyMapList.'.format(k+1, len(HDUidx)))
        hdulst.close()

    def view(self,**kwargs):
        pass

    def healpix(self, nside=HEALPIX_NSIDE, nest=HEALPIX_NEST, overwrite=True):
        """Project all members of the current SkyMapList to a HEALPix grid.
Parameters:
nside is used to define the HEALPix grid.
overwrite is a boolean option. When it is set to True, overlapped HEALPixel
is overwritten by latter assignment.
"""
        hpxcdata = np.zeros(hp.nside2npix(nside),dtype=np.double)
        hpxoccur = np.zeros(hp.nside2npix(nside),dtype=np.int64)
        for k in range(0,len(self)):
            assert isinstance(self[k], SkyMap)
            self[k].pprint()
            pix = self[k].grid.boundary.inside_hpx(nside)
            x,y,z = hp.pix2vec(nside, pix, nest=nest)
            xi,yi,_ = self[k].grid.locate(x,y,z)
            xi = np.array(np.round(xi),dtype=np.int64)
            yi = np.array(np.round(yi),dtype=np.int64)
            if overwrite:
                hpxcdata[pix] = self[k].cdata[yi,xi].astype(np.double)
                hpxoccur[pix] = 1
            else:
                hpxcdata[pix] = hpxcdata[pix] + self[k].cdata[yi,xi]
                hpxoccur[pix] = hpxoccur[pix] + 1
        return hpxcdata,hpxoccur

    def from_healpix(
            self,
            hpxdata,
            csys = DEFAULT_CSYS,
            M    = None,
            N    = NUM_PIX,
            span = PIXEL_GRID_SPAN,
            nest = HEALPIX_NEST):
        """Project HEALPix map to members of the current SkyMapList.
If the current SkyMapList is empty and essential parameters are present,
a series of empty SkyMap objects are initiated for projected HEALPixels,
which cover the whole celestial sphere all together.
Essential parameters to define closed celestial sphere coverage:
M is the number of SkyMap objects along the same meridian every 90 degrees.
thus there are 4*M SkyMap objects along each meridian.
N is the number of pixels along each axis of the pixel grid of each SkyMap.
span is the angular span along each axis of the pixel grid of each SkyMap.
Other parameter(s):
nest is a boolean option, which specifies if the pixel indices of input
HEALPix image are nested or not.
"""
        npix = len(hpxdata)
        print(u'{} healpixel loaded.'.format(npix))
        nside = hp.npix2nside(npix)
        if len(self) == 0 and M is not None:
            print(u'Create new SkyMap instances from celestial tessellation.')
            phi,theta = tessellangles(M)
            for k in range(len(phi)):
                self.append(SkyMap(
                    N     = N,
                    span  = span,
                    phi   = phi[k],
                    theta = theta[k],
                    psi   = 0.0,
                    csys  = csys
                ))
                print(u'SkyMap {}/{} created.'.format(k+1, len(phi)))
        for k in range(0,len(self)):
            assert isinstance(self[k], SkyMap)
            self[k].pprint()
            phi, theta = convert_csys(
                self[k].grid.phi, self[k].grid.theta, from_csys=self[k].csys, to_csys=hpxcsys)
            pix = hp.ang2pix(nside, np.pi/2.0-theta, phi, nest=nest)
            self[k].cdata = np.array(hpxdata[pix], dtype=np.double)

    def show2D(
            self,
            axes   = None,
            csys   = DEFAULT_CSYS,
            xlabel = r'$l$',
            ylabel = r'$b$',
            cmap   = cm.gray,
            npts   = 800):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        x,y,cmask = mollweide_canvas(npts)
        allskymap = np.zeros(x.shape)
        coverage  = np.zeros(x.shape)
        for i in range(len(self)):
            cdata, _, _, _, gmask = self[i].view(csys=csys, projection='mollweide', npts=npts)
            allskymap[gmask] = cdata[gmask]
            coverage[gmask] += 1.0
            print(u'{}/{} sky map loaded and rendered.'.format(i+1, len(self)))
        ma = np.ma.array(allskymap/np.clip(coverage, 1, None), mask=(coverage<1))
        axes.imshow(ma, origin='lower', extent=(-180, 180, -90, 90), cmap=cmap)
        axes.set_xlabel('{}, in deg'.format(xlabel))
        axes.set_ylabel('{}, in deg'.format(ylabel))
        if draw_now:
            plt.show()
        return axes

    def pprint(self):
        for m in self:
            m.pprint()

def ev2hpx(
        phi,
        theta,
        weights = None,
        memmap  = False,
        prefix  = None,
        dtype   = 'float64',
        nside   = HEALPIX_NSIDE,
        nest    = HEALPIX_NEST,
        verbose = False):
    """Make histogram on HEALPix grid from input events.
Use memmap for huge set of events or high-angular-resolution HEALPix
pixelisation.

Types of events:
1. Photon arrival. Unit: count. Photon arrivals are primary events
extracted from a set of observed data. Events of this type are additive.
Photon arrivals detected while exposure are effective, otherwise are
considered as background counts.
2. Pass-by. Unit: second. Abstract events occur at arbitrary given
frequency. Events of this type are addtive. Sum of all pass-by events
of a set of observed data yields the full operating time (in seconds)
of the observation.
3. Exposure. Unit: second. Abstract events occur at arbitrary given
frequency while the current detector is working. These events are
discrete subintervals of a good-time-interval (GTI) in temporal data
analysis. Sum of all exposure events extracted from a set of observed
data yields the net observation time, or, the exposure duration.

Each event is stamped with the time it occurs. From the timestamp we
can infer the location of the origin (for photon arrivals) or target
(for pass-by or exposure) of the event on the celestial sphere. The
location is thus called location of the event. Events are binned into
celestial pixels according to their locations. An image defined on
the celestial sphere with each of its values represents the count of
occurrences of events of a type is a 2-D histogram in this way,
represents the distribution of occurrences of events on the celestial
sphere, e.g., image of photon arrivals is the most intuitive image
represents the photon brightness distribution, image of exposure is
the image visualises the exposure distribution across the celestial.

Parameters:
phi and theta are angular coordinates of the events. phi is longitude
goes from -pi to pi while theta is latitude goes from -pi/2 (south pole)
to pi/2 (north pole).
weights are weights of the events. If not specified all events are
considered as equally weighted.
memmap is boolean option. If numpy.memmap is used for the histogram.
prefix is string. If memmap is used this is the prefix of names of
the memory map files.
dtype is the data-type used to interpret the histogram as well as the
memmap file contents.
nside is number of sides of HEALPix grid.
nest is boolean option. If the HEALPix order scheme is NESTED (if true)
or RING (if false).
"""
    nevs = len(phi)
    nbins = hp.nside2npix(nside)
    if verbose:
        print(u'{} events in total.'.format(nevs))
    assert hp.isnsideok(nside), '{} is not a valid NSIDE.'.format(nside)
    if verbose:
        print(u'HEALPix grid resolution: {} arcmin.'.format(hp.nside2resol(nside,arcmin=True)))
        print(u'{} MB used for the histogram.'.format(np.double(np.dtype(dtype).itemsize) * nbins/1024.0**2))
    if memmap:
        if prefix is None:
            prefix = path.join(mkdtemp(), 'hpx_')
        hist_file = prefix + 'hist.dat'
        evpix_file = prefix + 'evpix.dat'
        histcts = np.memmap(hist_file, dtype=dtype, mode='w+',shape=nbins)
        evpix = np.memmap(evpix_file, dtype='int64', mode='w+',shape=nbins)
    else:
        histcts = np.zeros(nbins,dtype=dtype)
        evpix = np.zeros(nbins,dtype='int64')
    evpix = np.ravel(hp.ang2pix(nside,np.pi/2.0-theta,phi,nest=nest))
    histcts[:] = np.bincount(evpix,weights=weights,minlength=nbins)
    return histcts

def parse_angle(val):
    """Parse anglular expression into float number, in radian.
Accepted units:
  rad - radian
  deg - degree
  arcmin - arc minute
  arcsec - arc second
"""
    if val.lower().endswith('rad'):
        span = np.double(val[:-3])
    elif val.lower().endswith('deg'):
        span = np.deg2rad(np.double(val[:-3]))
    elif val.lower().endswith('arcmin'):
        span = np.deg2rad(np.double(val[:-6])/60.0)
    elif val.lower().endswith('arcsec'):
        span = np.deg2rad(np.double(val[:-6])/3600.0)
    else:
        span = np.double(val)
    return span

def main():
    opts, args = gnu_getopt(
        sys.argv[1:],
        'c:s:N:M:i:o:q:O:',
        [
            'csys=',
            'span='
            'resolution=',
            'res==',
            'tessellation=',
            'tes==',
            'out==',
            'output=',
            'in==',
            'input=',
            'quaternion=',
            'quat=',
            'angles=',
            'axis=',
            'up=',
            'hdu=',
            'allsky',
            'overwrite',
        ])
    action    = args[0]
    csys      = DEFAULT_CSYS
    span      = np.pi/2
    N         = 512
    quat      = None
    phi       = None
    theta     = None
    psi       = None
    axis      = None
    up        = None
    M         = None
    hduidx    = 0
    allsky    = False
    overwrite = False
    for opt, val in opts:
        if opt in ['-c', '--csys']:
            csys = val
        elif opt in ['-s', '--span']:
            if ',' in val:
                span = tuple(map(parse_angle, val.split(',')))
            else:
                span = parse_angle(val)
        elif opt in ['-N', '--res', '--resolution']:
            if ',' in val:
                N = tuple(map(int, val.split(',')))
            else:
                N = int(val)
        elif opt in ['-M', '--tes', '--tessellation']:
            allsky = True
            M = int(val)
        elif opt in ['-i', '--in', '--input']:
            inputfile = path.normpath(path.abspath(path.realpath(val)))
            assert path.isfile(inputfile), 'input file does not exist.'
        elif opt in ['-o', '--out', '--output']:
            outputfile = path.normpath(path.abspath(path.realpath(val)))
        elif opt in ['-q', '--quat', '--quaternion']:
            quat = np.double(list(map(np.double, val.split(','))))
        elif opt in ['--angles']:
            phi, theta, psi = map(parse_angle, val.split(','))
        elif opt in ['--axis']:
            axis = np.double(list(map(np.double, val.split(','))))
        elif opt in ['--up']:
            up = np.double(list(map(np.double, val.split(','))))
        elif opt in ['--allsky']:
            allsky = True
        elif opt in ['--overwrite']:
            overwrite = True
        elif opt in ['-O']:
            overwrite = True
            outputfile = path.normpath(path.abspath(path.realpath(val)))
        elif opt.lower() in ['--hdu']:
            hduidx = int(val)
    if action.lower() == 'help':
        print(__doc__)
    elif action.lower() == 'init':
        if allsky:
            sms = SkyMapList(M=M, N=N, span=span, csys=csys)
            sms.save(outputfile, overwrite=overwrite)
        else:
            sm  = SkyMap(
                N     = N,
                csys  = csys,
                span  = span,
                quat  = quat,
                axis  = axis,
                up    = up,
                phi   = phi,
                theta = theta,
                psi   = psi
            )
            sm.save(outputfile, overwrite=overwrite)
    elif action.lower() == 'print':
        if allsky:
            sms = SkyMapList()
            sms.load(inputfile)
            sms.pprint()
        else:
            sm = SkyMap(fitsfile=inputfile, HDUidx=hduidx)
            sm.pprint()
    elif action.lower() == 'show':
        if allsky:
            sms = SkyMapList()
            sms.load(inputfile)
            sms.show2D()
        else:
            sm = SkyMap(fitsfile=inputfile, HDUidx=hduidx)
            sm.show2D()
if __name__ == '__main__':
    main()
