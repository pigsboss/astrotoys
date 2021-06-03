#!/usr/bin/env python
#coding=utf-8
"""Planar Projection Pixel Grid (p3g).
Copyright: pigsboss@github
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymath.quaternion as quaternion
import pymath.sphere as sphere
from pymath.common import xyz2ptr, ptr2xyz, norm, direction, as_xyz, spherical_rectangle_area
from astrotoys.wcstime import mollweide, imollweide
from mpl_toolkits.mplot3d import Axes3D

NUM_PIX         = 512
NUM_PTS         = 100
NUM_LON_GRIDS   = 5
NUM_LAT_GRIDS   = 5
PIXEL_GRID_SPAN = np.deg2rad(24.0)
HEALPIX_NSIDE   = 128
HEALPIX_NEST    = True

def plot_globe_grids(
        axes      = None,
        nlons     = NUM_LON_GRIDS,
        nlats     = NUM_LAT_GRIDS,
        npts      = NUM_PTS,
        linespecs = 'k-',
        linewidth = 0.5):
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        draw_now = True
    else:
        draw_now = False
    pgv = np.arange(nlons*2)/nlons*np.pi
    tgv = np.arange(npts)/(npts-1)*np.pi-np.pi/2
    gx, gy, gz = ptr2xyz(*tuple(np.meshgrid(pgv, tgv)))
    for i in range(2*nlons):
        axes.plot(gx[:,i], gy[:,i], gz[:,i], linespecs, linewidth=linewidth)
    pgv = np.arange(npts)/(npts-1)*2*np.pi
    tgv = np.arange(1, nlats+1)/(nlats+1)*np.pi-np.pi/2
    gx, gy, gz = ptr2xyz(*tuple(np.meshgrid(pgv, tgv)))
    for i in range(nlats):
        axes.plot(gx[i,:], gy[i,:], gz[i,:], linespecs, linewidth=linewidth)
    if draw_now:
        plt.show()
    return axes

def tessellangles(M):
    """Generate angles of pixel grids that cover the whole sphere.
"""
    phi = np.zeros(16*M*M)
    theta = np.zeros(16*M*M)
    k = 0
    # Northern polar tessella
    phi[k] = 0
    theta[k] = np.pi/2.0
    # Southern polar tessella
    k = k+1
    phi[k] = 0
    theta[k] = -np.pi/2.0
    # Equatorial tessellae
    for m in range(0,4*M):
        k = k+1
        phi[k] = 0.5*m*np.pi/M
        theta[k] = 0
    # Others
    for m in range(1,M):
        phiinc = 2.0*np.arctan(np.tan(np.pi/4.0/M)/(np.cos(m*np.pi/2.0/M)+\
            np.sin(m*np.pi/2.0/M)*np.tan(np.pi/4.0/M)))
        N = np.int(np.ceil(2.0*np.pi/phiinc))
        for n in range(0,N):
            k = k+1
            phi[k] = n*phiinc
            theta[k] = 0.5*m*np.pi/M
            k = k+1
            phi[k] = n*phiinc
            theta[k] = -0.5*m*np.pi/M
    return phi[0:k+1], theta[0:k+1]

class Particles:
    """Particles in 3D space.
"""
    def __init__(self, a, b, c, csys='cartesian'):
        if (csys.lower() == 'cartesian') | (csys.lower() == 'xyz'):
            self.xyz = np.array([a,b,c])
            self.x = np.double(self.xyz[0])
            self.y = np.double(self.xyz[1])
            self.z = np.double(self.xyz[2])
            self.phi, self.theta, self.rho = xyz2ptr(self.x, self.y, self.z)
        elif (csys.lower() == 'polar') | (csys.lower() == 'sphere') | (csys.lower() == 'spherical'):
            self.phi   = np.double(a)
            self.theta = np.double(b)
            self.rho   = np.double(c)
            self.xyz = np.array(ptr2xyz(self.phi, self.theta, self.rho))
            self.x = np.double(self.xyz[0])
            self.y = np.double(self.xyz[1])
            self.z = np.double(self.xyz[2])
        else:
            assert False, 'unsupported coordinate system.'

    def __set_angles__(self,phi,theta):
        self.phi = np.double(phi)
        self.theta = np.double(theta)
        self.xyz = np.array(ptr2xyz(self.phi, self.theta, self.rho))
        self.x = np.double(self.xyz[0])
        self.y = np.double(self.xyz[1])
        self.z = np.double(self.xyz[2])

    def __set_xyz__(self,x,y,z):
        self.xyz = np.array([x,y,z])
        self.x = np.double(self.xyz[0])
        self.y = np.double(self.xyz[1])
        self.z = np.double(self.xyz[2])
        self.phi, self.theta, self.rho = xyz2ptr(self.x, self.y, self.z)

    def __set_ptr__(self,phi,theta,rho):
        self.phi = np.double(a)
        self.theta = np.double(b)
        self.rho = np.double(c)
        self.xyz = np.array(ptr2xyz(self.phi, self.theta, self.rho))
        self.x = np.double(self.xyz[0])
        self.y = np.double(self.xyz[1])
        self.z = np.double(self.xyz[2])

    def rotate(self,q):
        r = quaternion.rotate(quat=q, \
            vector=np.array([np.array(self.x).reshape(-1), \
            np.array(self.y).reshape(-1), \
            np.array(self.z).reshape(-1)]))
        self.__set_xyz__(np.array(r[0,:]).reshape(self.x.shape), \
            np.array(r[1,:]).reshape(self.x.shape), \
            np.array(r[2,:]).reshape(self.x.shape))

    def show2D(
            self,
            axes           = None,
            local_only     = True,
            projection     = 'mollweide',
            sc_color       = 'b',
            sc_size        = 20,
            sc_marker      = '.',
            canvas_margin  = 0.2,
            grid_on        = True,
            grid_linespecs = 'k--',
            grid_linewidth = 0.5,
            grid_nlats     = NUM_LAT_GRIDS,
            grid_nlons     = NUM_LON_GRIDS,
            grid_npts      = NUM_PTS,
            xunit          = 'rad',
            yunit          = 'rad',
            xlabel         = 'Longitude',
            ylabel         = 'Latitude',
            axes_fontsize  = None):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        # convert unit
        if xunit.lower().startswith('r'):
            xfactor = 1.0
        elif xunit.lower().startswith('d'):
            xfactor = 180.0/np.pi
        elif xunit.lower().startswith('a'): # arcsec
            xfactor = 180.0*3600.0/np.pi
        if yunit.lower().startswith('r'):
            yfactor = 1.0
        elif yunit.lower().startswith('d'):
            yfactor = 180.0/np.pi
        elif yunit.lower().startswith('a'): # arcsec
            yfactor = 180.0*3600.0/np.pi
        # show 2D scatter diagram
        if projection.lower().startswith('m'):
            sx, sy = mollweide(self.phi, self.theta)
        elif projection.lower().startswith('c'):
            sx, sy = self.phi, self.theta
        axes.scatter(sx*xfactor, sy*yfactor, c=sc_color, s=sc_size, marker=sc_marker)
        # draw grids
        if grid_on:
            if local_only:
                ## find minimum closure of scatters
                phi_min   = max(-np.pi,      0.5*(self.phi.min()   + self.phi.max())   - (1+canvas_margin)*0.5*(self.phi.max()   - self.phi.min()))
                phi_max   = min( np.pi,      0.5*(self.phi.min()   + self.phi.max())   + (1+canvas_margin)*0.5*(self.phi.max()   - self.phi.min()))
                theta_min = max(-np.pi/2.0,  0.5*(self.theta.min() + self.theta.max()) - (1+canvas_margin)*0.5*(self.theta.max() - self.theta.min()))
                theta_max = min( np.pi/2.0,  0.5*(self.theta.min() + self.theta.max()) + (1+canvas_margin)*0.5*(self.theta.max() - self.theta.min()))
            else:
                phi_min, phi_max = -np.pi, np.pi
                theta_min, theta_max = -np.pi/2, np.pi/2
            ## generate grids
            gx  = np.zeros((grid_npts, grid_nlons+grid_nlats))
            gy  = np.zeros((grid_npts, grid_nlons+grid_nlats))
            xgv = np.arange(grid_nlons)/(grid_nlons-1) * (  phi_max-  phi_min) +   phi_min
            ygv = np.arange(grid_npts) /(grid_npts -1) * (theta_max-theta_min) + theta_min
            gx[:,:grid_nlons], gy[:,:grid_nlons] = np.meshgrid(xgv, ygv)
            xgv = np.arange(grid_nlats)/(grid_nlats-1) * (theta_max-theta_min) + theta_min
            ygv = np.arange(grid_npts )/(grid_npts -1) * (  phi_max-  phi_min) +   phi_min
            gy[:,grid_nlons:], gx[:,grid_nlons:] = np.meshgrid(xgv, ygv)
            if projection.lower().startswith('m'):
                gx, gy = mollweide(gx, gy)
            axes.plot(gx*xfactor, gy*yfactor, grid_linespecs, linewidth=grid_linewidth)
        # adjust axis
        axes.set_xlabel('{}, in {}.'.format(xlabel, xunit), fontsize=axes_fontsize)
        axes.set_ylabel('{}, in {}.'.format(ylabel, yunit), fontsize=axes_fontsize)
        if draw_now:
            plt.show()
        return axes

    def show3D(
            self,
            axes           = None,
            local_only     = True,
            sc_color       = 'b',
            sc_marker      = '.',
            sc_size        = 20,
            grid_on        = True,
            grid_nlons     = NUM_LON_GRIDS,
            grid_nlats     = NUM_LAT_GRIDS,
            grid_npts      = NUM_PTS,
            grid_linespecs = 'k-',
            grid_linewidth = 0.5,
            axes_fontsize  = None,
            xlabel         = r'$x$',
            ylabel         = r'$y$',
            zlabel         = r'$z$'):
        if axes is None:
            fig = plt.figure()
            axes = fig.gca(projection='3d')
            draw_now = True
        else:
            draw_now = False
        if not local_only:
            if grid_on:
                plot_globe_grids(
                    axes      = axes,
                    nlons     = grid_nlons,
                    nlats     = grid_nlats,
                    npts      = grid_npts,
                    linespecs = grid_linespecs,
                    linewidth = grid_linewidth)
        axes.scatter(self.x, self.y, self.z, s=sc_size, c=sc_color, marker=sc_marker)
        axes.set_xlabel(xlabel, fontsize=axes_fontsize)
        axes.set_ylabel(ylabel, fontsize=axes_fontsize)
        axes.set_zlabel(zlabel, fontsize=axes_fontsize)
        if draw_now:
            plt.show()
        return axes

class ParticlesOnUnitSphere(Particles):
    def __init__(self, phi, theta):
        if phi is not None:
            if np.double(phi).ndim == 0:
                rho = 1.0
            else:
                rho = np.ones(phi.shape)
            super(ParticlesOnUnitSphere,self).__init__(phi,theta,rho,csys='polar')

class RandomParticlesOnUnitSphere(ParticlesOnUnitSphere):
    """Particles distributed uniformly on unit sphere.
"""
    def __init__(self, npts=NUM_PTS):
        phi = np.random.rand(npts)*2.0*np.pi - np.pi
        theta = np.arcsin(np.random.rand(npts)*2.0 - 1.0)
        super(RandomParticlesOnUnitSphere,self).__init__(phi,theta)

class RigidBody3D(Particles):
    def __init__(
            self,
            axis  = None,
            up    = None,
            quat  = None,
            phi   = None,
            theta = None,
            psi   = None):
        quat, axis, up, phi, theta, psi, _ = quaternion.fit_attitude(quat, axis, up, phi, theta, psi)
        Particles.__init__(self, *tuple(np.transpose(np.vstack((axis,up)))), csys='xyz')
        self.__update__()
    def __update__(self):
        self.axis = self.xyz[:,0]
        self.up = self.xyz[:,1]
        self.quat = quaternion.from_axes(self.axis,self.up)
    def rotate(self,quat):
        Particles.rotate(self,quat)
        self.__update__()
    def angles(self):
        return quaternion.angles(self.quat)

class SphericalPolygon(ParticlesOnUnitSphere):
    def __init__(self, vertices, external=None, npts=NUM_PTS):
        v = direction(as_xyz(vertices))
        nvs = v.shape[1]
        xyz = np.zeros([3, nvs * npts + 2 * nvs + 1])
        xyz[:, :nvs] = np.array(v)
        xyz[:, range(nvs, nvs*2)] = np.cross(v[:, range(-1, nvs-1)], v, axis=0)
        self.side_lens = sphere.distance(v[:, range(-1, nvs-1)], v)
        for i in range(0,nvs):
            xyz[:, range(nvs*2+npts*i, nvs*2+npts*(i+1))] = np.array(sphere.arc(v[:,np.mod(i-1,nvs)], v[:,np.mod(i,nvs)],npts=npts))
        self.nsides = nvs
        self.nvs = nvs
        self.npts = npts
        if external is None:
            ctr_phi, ctr_theta, ctr_rho = xyz2ptr(v[0,:].mean(), v[1,:].mean(), v[2,:].mean())
            cx,cy,cz = ptr2xyz(ctr_phi,ctr_theta,1.0)
            external = -1.0 * np.array([cx,cy,cz])
        xyz[:, nvs*npts+2*nvs] = np.array([external])
        ParticlesOnUnitSphere.__set_xyz__(self,xyz[0,:],xyz[1,:],xyz[2,:])

    def vertices(self):
        return np.array([self.x[0:self.nvs], self.y[0:self.nvs], self.z[0:self.nvs]])

    def sides(self):
        return np.array([self.x[range(self.nvs*2, self.nvs*(2+self.npts))], self.y[range(self.nvs*2, self.nvs*(2+self.npts))], self.z[range(self.nvs*2, self.nvs*(2+self.npts))]])

    def side_norms(self):
        return np.array([self.x[range(self.nvs,self.nvs*2)], self.y[range(self.nvs,self.nvs*2)], self.z[range(self.nvs,self.nvs*2)]])

    def external(self):
        """Point indicates exterior side of the sphrerical polygon.
"""
        return np.array([self.x[self.nvs*2+self.npts*self.nvs], self.y[self.nvs*2+self.npts*self.nvs], self.z[self.nvs*2+self.npts*self.nvs]])

    def __intersections__(self,a,b):
        a = as_xyz(a)
        b = as_xyz(b)
        d = sphere.distance(a, b)
        norm_ab = sphere.axis(a, b)
        p = []
        ipts = []
        crossed = []
        delta = 10.0 * DEPS
        for i in range(self.nvs):
            q0 = sphere.axis(norm_ab, self.side_norms()[:,i])
            q1 = -1.0 * q0
            p.append(q0)
            p.append(q1)
            q0a = sphere.distance(q0, a)
            q0b = sphere.distance(q0, b)
            q0c = sphere.distance(q0, self.vertices()[:,np.mod(i-1,self.nvs)])
            q0d = sphere.distance(q0, self.vertices()[:,np.mod(i,self.nvs)])
            q1a = sphere.distance(q1, a)
            q1b = sphere.distance(q1, b)
            q1c = sphere.distance(q1, self.vertices()[:,np.mod(i-1,self.nvs)])
            q1d = sphere.distance(q1, self.vertices()[:,np.mod(i,self.nvs)])
            q0_inter = (np.abs(q0a+q0b-d)<=delta) & (np.abs(q0c+q0d-self.side_lens[i])<=delta)
            q1_inter = (np.abs(q1a+q1b-d)<=delta) & (np.abs(q1c+q1d-self.side_lens[i])<=delta)
            if q0_inter:
                if ipts == []:
                    ipts.append(q0)
                    crossed.append(True)
                elif (sphere.distance(q0,np.array(ipts).T) > delta).all():
                    ipts.append(q0)
                    crossed.append(True)
            if q1_inter:
                if ipts == []:
                    ipts.append(q1)
                    crossed.append(True)
                elif (sphere.distance(q1,np.array(ipts).T) > delta).all():
                    ipts.append(q1)
                    crossed.append(True)
        return np.array(ipts).T, crossed, np.array(p).T

    def inside(self, p):
        p = np.array(p)
        if p.ndim == 1:
            p = np.array([p]).T
            npts = 1
        else:
            ndim,npts = p.shape
        t = []
        for i in range(0,npts):
            ipts,f,r = self.__intersections__(self.external(), p[:,i])
            t.append(np.mod(np.int32(np.array(f)).sum(),2) != 0)
        return np.squeeze(np.array(t))

    def show2D(
            self,
            axes           = None,
            projection     = 'mollweide',
            xunit          = 'rad',
            yunit          = 'rad',
            side_linespecs = 'b-',
            side_linewidth = 1.0,
            **kwargs):
        # parse input arguments
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        if xunit.lower().startswith('r'):
            xfactor = 1.0
        elif xunit.lower().startswith('d'):
            xfactor = 180.0/np.pi
        elif xunit.lower().startswith('a'): # arcsec
            xfactor = 180.0*3600.0/np.pi
        if yunit.lower().startswith('r'):
            yfactor = 1.0
        elif yunit.lower().startswith('d'):
            yfactor = 180.0/np.pi
        elif yunit.lower().startswith('a'): # arcsec
            yfactor = 180.0*3600.0/np.pi
        Vertices = Particles(*tuple(self.vertices()))
        Vertices.show2D(axes=axes, projection=projection, xunit=xunit, yunit=yunit, **kwargs)
        phi,theta,rho = xyz2ptr(*tuple(self.sides()))
        if projection.lower().startswith('m'):
            lx,ly = mollweide(phi,theta)
        elif projection.lower().startswith('c'):
            lx,ly = phi,theta
        else:
            assert False, 'unrecognized projection {}'.format(projection)
        axes.plot(lx*xfactor, ly*yfactor, side_linespecs, linewidth=side_linewidth)
        if draw_now:
            plt.show()
        return axes

    def show3D(
            self,
            axes           = None,
            side_linespecs = 'b-',
            side_linewidth = 1.0,
            **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            draw_now = True
        else:
            draw_now = False
        s_xyz = self.sides()
        axes.plot(s_xyz[0], s_xyz[1], s_xyz[2], side_linespecs, linewidth=side_linewidth)
        Vertices = Particles(*tuple(self.vertices()))
        Vertices.show3D(axes=axes, **kwargs)
        if draw_now:
            plt.show()
        return axes

class SphericalRectangle(SphericalPolygon):
    def __init__(self, vertices, external=None, npts=NUM_PTS):
        SphericalPolygon.__init__(self, vertices, external, npts)
        assert self.nvs == 4, 'spherical rectangle must have exactly 4 vertices.'
        pc =      self.vertices().mean(axis=1)               # center of the secant plane defined by the vertices of the spherical rectangle.
        pn = 0.5*(self.vertices()[:,2]+self.vertices()[:,3]) # center of the north (top) side of the planar rectangle (the projection onto the secant plane).
        ps = 0.5*(self.vertices()[:,0]+self.vertices()[:,1]) # center of the sourth (bottom) side of the planar rectangle.
        pw = 0.5*(self.vertices()[:,0]+self.vertices()[:,3]) # center of the west (left) side.
        pe = 0.5*(self.vertices()[:,1]+self.vertices()[:,2]) # center of the east (right) side.
        self.axis   = direction(pc)                            # direction (unit vector) from the orgin to the center of the rectangle (AXIS-vector).
        self.up     = direction(pn - ps)                       # direction (unit vector) from the south to the north of the rectangle (UP-vector).
        self.quat   = quaternion.from_axes(axis=self.axis, up=self.up)
        self.size_u = norm(pe - pw) / norm(pc)
        self.span_u = np.arctan(self.size_u / 2.0) * 2.0
        self.size_v = norm(pn - ps) / norm(pc)
        self.span_v = np.arctan(self.size_v / 2.0) * 2.0
        self.size   = (self.size_v, self.size_u)
        self.span   = (self.span_v, self.span_u)
        self.area   = spherical_rectangle_area(self.span_v, self.span_u)
        self.solid_angle = self.area

    def rotate(self, quat):
        Particles.rotate(self, quat)
        self.axis = quaternion.rotate(quat=quat, vector=self.axis)
        self.up   = quaternion.rotate(quat=quat, vector=self.up)
        self.quat = quaternion.multiply(quat, self.quat)

    def inside(self, p):
        """Find if p is inside the rectangle.
"""
        pref = quaternion.rotate(quaternion.conjugate(self.quat), p)
        return (np.abs(pref[1]) <= np.abs(self.size_v*0.5*pref[0])) & \
               (np.abs(pref[2]) <= np.abs(self.size_u*0.5*pref[0])) & \
               (pref[0]>0)

    def inside_hpx(self, nside):
        """Find HEALPix pixels (NESTED scheme) inside the rectangle.
"""
        assert nside>0, 'nside must be a positive integer.'
        assert np.mod(np.log2(nside), 1) == 0, 'nside must be power of 2.'
        jmin = int(np.clip(np.ceil(np.max([np.log2(np.sqrt(4.0*np.pi/self.area/12.0)),0])),4,None))
        pidxlst = range(12*(4**(jmin-1))) # previous indices list
        ploclst = [False]*len(pidxlst)    # previous locations list. True: inside; False: on the boundary.
        J = np.int(np.log2(nside))
        for j in range(jmin, J+1):
            cidxlst = [] # current indices list
            cloclst = [] # current locations list
            for i in range(len(pidxlst)):
                if ploclst[i]:
                    cloclst.extend([ploclst[i]]*4)
                    cidxlst.extend(pidxlst[i]*4+np.arange(4))
                else:
                    for k in range(4):
                        pix = pidxlst[i]*4 + k
                        pcs = hp.boundaries(2**j,pix,2**(J-j),nest=True)
                        v = np.concatenate((np.reshape(np.double(hp.pix2vec(2**j,\
                            pix,nest=True)),(3,1)),pcs),axis=1)
                        ntps = 4*(2**(J-j)) + 1 # number of total points
                        nps  = np.sum(self.inside(v)) # number of inside points
                        if nps == ntps: # all points are inside
                            cloclst.append(True)
                            cidxlst.append(pix)
                        elif nps > 0:
                            cloclst.append(False)
                            cidxlst.append(pix)
            pidxlst = cidxlst # update indices list
            ploclst = cloclst # update locations list
        idxlst = np.array(cidxlst,dtype=np.int64)
        loclst = np.array(cloclst,dtype=np.bool)
        return idxlst[loclst]

class PixelGrid(ParticlesOnUnitSphere):
    def __init__(
            self,
            quat  = None,
            span  = PIXEL_GRID_SPAN,
            N     = NUM_PIX,
            axis  = None,
            up    = None,
            phi   = None,
            theta = None,
            psi   = None):
        """A pixel grid is specified by its local structure as well as its orientation
in the physical coordinate system.
Its local structure, more specifically, refers to its span and granularity.
Since a pixel grid is considered as a rigid body, its orientation is determined by two
of its intrinsic vectors, the AXIS vector and the UP vector. The AXIS vector is a unit
vector from the origin of the physical coordinate system to the center of the grid, while
the UP vector is the vertical axis vector of the image coordinate system defined on the
tangent plane of the unit sphere at center of the pixel grid.
The span of a pixel grid is measured by the geodesics of central column and row of the
pixel grid.
The granularity of a pixel grid is measured by number of pixels both column-wise and
row-wise.

Arguments that specify the local structure:
span - scalar or tuple (span_v, span_u), where v-axis is the vertical axis of the image
       coordinate system and u-axis is the horizontal one.
N    - scalar or tuple (NV, NU).

Arguments that specify the orientation:
axis and up specify the orientation of the pixel grid directly.

phi, theta and psi (Euler's angles) rotates the pixel grid from its initial orientation
to its current orientation.

quat is the quaternion rotates the pixel grid from its initial orientation to its current
orientation, or, equivalently, convert its coordinate from the physical coordinate system
to its intrinsic coordinate system.
"""
        quat,axis,up,phi,theta,psi,status = quaternion.fit_attitude(
            quat=quat, axis=axis, up=up, phi=phi, theta=theta, psi=psi)
        try:
            NV,NU = N
            if np.isscalar(span):
                span = (span, span)
        except:
            NV = N
            NU = N
        x = np.ones([NV,NU])
        if np.isscalar(span):
            a       = 2.0*np.tan(span / 2.0)
            au      = a
            av      = a
            px_sz   = a / np.double(N)
            px_sz_u = px_sz
            px_sz_v = px_sz
            span_u  = span
            span_v  = span
        else:
            a       = 2.0*np.tan(np.double(span) / 2.0)
            au      = a[1]
            av      = a[0]
            px_sz   = a / np.double(N)
            px_sz_u = px_sz[1]
            px_sz_v = px_sz[0]
            span_u  = span[1]
            span_v  = span[0]
        ygv = (np.double(range(0,NU))-(np.double(NU)-1.0)*0.5)/ \
            np.double(NU) * au
        zgv = (np.double(range(0,NV))-(np.double(NV)-1.0)*0.5)/ \
            np.double(NV) * av
        y,z = np.meshgrid(ygv,zgv)
        xyz = quaternion.rotate(quat = quat, \
            vector = direction(np.array([x,y,z])))
        v_xyz = quaternion.rotate(quat = quat, \
            vector = direction(np.array([ \
                [ 1.0,     1.0,    1.0,     1.0], \
                [-0.5*au,  0.5*au, 0.5*au, -0.5*au], \
                [-0.5*av, -0.5*av, 0.5*av,  0.5*av]])))
        Particles.__set_xyz__(self,xyz[0],xyz[1],xyz[2])
        self.grid_vertices = v_xyz
        self.boundary      = SphericalRectangle(v_xyz)
        self.axis          = axis
        self.up            = up
        self.psi           = psi
        self.quat          = np.array(np.double(quat) / np.sqrt((quat**2.0).sum()))
        self.grid_sz       = a
        self.grid_sz_u     = au
        self.grid_sz_v     = av
        self.px_sz         = px_sz   # pixel step size in tangential plane.
        self.px_sz_u       = px_sz_u
        self.px_sz_v       = px_sz_v
        self.span          = span    # span angle on unit sphere.
        self.span_u        = span_u
        self.span_v        = span_v
        self.N             = N
        self.NU            = NU
        self.NV            = NV
        self.u             = ygv
        self.v             = zgv
        self.px_area       = PixelGrid.__get_pixel_area__(self) # solid angles of each pixels, in steradian.

    def __locate_search__(self,phi,theta):
        idx,d = sphere.nearest(self.phi,self.theta,phi,theta)
        return idx,d

    def __locate_projection__(self,p):
        pref = quaternion.rotate(quaternion.conjugate(self.quat),p)
        p_inside = (np.abs(pref[1]) <= np.abs(self.grid_sz_u*0.5*pref[0])) & \
            (np.abs(pref[2]) <= np.abs(self.grid_sz_v*0.5*pref[0])) & \
            (pref[0]>0)
        if np.isscalar(p[0]):
            if p_inside:
                xidx = pref[1] / pref[0] / self.px_sz_u + \
                    (self.NU - 1.0)*0.5
                yidx = pref[2] / pref[0] / self.px_sz_v + \
                    (self.NV - 1.0)*0.5
            else:
                xidx = np.nan
                yidx = np.nan
        else:
            xidx = np.zeros(p[0].shape)
            yidx = np.zeros(p[0].shape)
            xidx[~p_inside] = np.nan
            yidx[~p_inside] = np.nan
            xidx[p_inside] = pref[1][p_inside] / pref[0][p_inside] / self.px_sz_u + \
                (self.NU - 1.0)*0.5
            yidx[p_inside] = pref[2][p_inside] / pref[0][p_inside] / self.px_sz_v + \
                (self.NV - 1.0)*0.5
        return xidx,yidx,p_inside

    def pixel_vertices(self):
        """Get pixel vertices.
A pixel is an spherical area surrounded by a spherical square.
This function returns 4 vertices of each pixel on the grid.
"""
        x = np.ones([self.NV,self.NU])
        y, z = np.meshgrid(self.u, self.v)
        v0 = quaternion.rotate(self.quat,\
            direction(np.array([x,y-self.px_sz_u*0.5,z-self.px_sz_v*0.5])))
        v1 = quaternion.rotate(self.quat,\
            direction(np.array([x,y+self.px_sz_u*0.5,z-self.px_sz_v*0.5])))
        v2 = quaternion.rotate(self.quat,\
            direction(np.array([x,y+self.px_sz_u*0.5,z+self.px_sz_v*0.5])))
        v3 = quaternion.rotate(self.quat,\
            direction(np.array([x,y-self.px_sz_u*0.5,z+self.px_sz_v*0.5])))
        return v0,v1,v2,v3

    def __get_pixel_area__(self):
        """Find solid angles of each pixels, in steradian.
"""
        x = np.ones([self.NV,self.NU])
        y, z = np.meshgrid(self.u, self.v)
        v0 = direction(np.array([x,y-self.px_sz_u*0.5,z-self.px_sz_v*0.5]))
        v1 = direction(np.array([x,y+self.px_sz_u*0.5,z-self.px_sz_v*0.5]))
        v2 = direction(np.array([x,y+self.px_sz_u*0.5,z+self.px_sz_v*0.5]))
        v3 = direction(np.array([x,y-self.px_sz_u*0.5,z+self.px_sz_v*0.5]))
        d02 = sphere.distance(v0,v2)
        d13 = sphere.distance(v1,v3)
        return d02*d13*0.5

    def rotate(self,quat):
        Particles.rotate(self,quat)
        self.axis = quaternion.rotate(quat=quat, vector=self.axis)
        self.up = quaternion.rotate(quat=quat, vector=self.up)
        self.quat = quaternion.multiply(quat, self.quat)
        self.grid_vertices = quaternion.rotate(quat=quat, \
            vector = self.grid_vertices)
        self.boundary.rotate(quat)

    def locate(self,*args):
        if len(args) == 2:
            p = np.array(ptr2xyz(args[0],args[1],1.0))
        elif len(args) == 3:
            p = np.array(args,dtype=np.double)
        return PixelGrid.__locate_projection__(self,p)

    def ang2pix(self,phi,theta):
        p = np.array(ptr2xyz(args[0],args[1],1.0))
        return PixelGrid.__locate_projection__(self,p)

    def vec2pix(self,x,y,z):
        p = np.array([x,y,z],dtype=np.double)
        return PixelGrid.__locate_projection__(self,p)

    def show2D(self, axes=None, grid_on=True, boundary_on=False, **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        if boundary_on:
            self.boundary.show2D(axes=axes, grid_on=grid_on, **kwargs)
            grid_on = False
        Particles.show2D(self, axes=axes, grid_on=grid_on, **kwargs)
        if draw_now:
            plt.show()

    def show3D(self, axes=None, **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            draw_now = True
        else:
            draw_now = False
        ParticlesOnUnitSphere.show3D(self, axes=axes, **kwargs)
        SphericalPolygon.show3D(self.boundary, axes=axes)
        if draw_now:
            plt.show()

    def info(self):
        """alias of pprint.
"""
        self.pprint()

    def pprint(self):
        """Pretty print.
"""
        print('{:<20}:[{:< 6.3f}, {:< 6.3f}, {:< 6.3f}, {:< 6.3f}]'.format('quaternion', *self.quat))
        print('{:<20}:[{:< 6.3f}, {:< 6.3f}, {:< 6.3f}]'.format('pointing axis', *self.axis))
        print('{:<20}:[{:< 6.3f}, {:< 6.3f}, {:< 6.3f}]'.format('position axis', *self.up))
        print('{:<20}:{:> 8.3f}, {:> 8.3f}, {:> 8.3f}'.format('proper euler angles', *np.rad2deg(quaternion.angles(self.quat))))
        print('{:<20}:{:> 6,d} (W) x {:> 6,d} (H)'.format('number of pixels', self.NU, self.NV))
        print('{:<20}:{:> 6.3f} deg. (W) x {:> 6.3f} deg. (H)'.format('pixel grid span', np.rad2deg(self.span_u), np.rad2deg(self.span_v)))
        print('{:<20}:{:> 6.3f} square-degrees'.format('pixel grid area', self.area()*(180.0/np.pi)**2))
        print('{:<20}:{:> 6.1f} arcssec (W) x {:> 6.1f} arcsec (H)'.format('pixel size', np.rad2deg(self.px_sz_u)*3600.0, np.rad2deg(self.px_sz_v)*3600.0))

    def area(self):
        """Area (solid angle) of pixel grid, in steradian.
"""
        return spherical_rectangle_area(self.span_u, self.span_v)

class PixelGridList(list):
    def __init__(self, M=None, N=NUM_PIX, span=PIXEL_GRID_SPAN):
        if M is None:
            if np.iterable(args) == 1:
                if len(args) > 0:
                    if isinstance(args[0],PixelGrid):
                        list.__init__(self,args)
                    else:
                        raise StandardError(str(args[0]) + ' is not PixelGrid object.')
                else:
                    list.__init__(self)
            else:
                list.__init__(self)
        else:
            list.__init__(self)
            print('Generate ' + str(M) + ':')
            if span < np.pi*0.5/M:
                print('Pixel grid span is less than the maximum interval.')
            phi,theta = tessellangles(M)
            for k in range(0,len(phi)):
                print(u'{} grid: phi={}, theta={}, N={}, span={}.'.format(
                    k, phi[k], theta[k], N, span
                ))
                self.append(PixelGrid(N=N,span=span,phi=phi[k],theta=theta[k],psi=0.0))
    def pprint(self):
        for g in self:
            g.pprint()
    def show2D(self, axes=None, **kwargs):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, aspect='equal')
            draw_now = True
        else:
            draw_now = False
        c = 'br'
        t = 0
        for g in self:
            g.show2D(axes, sc_color=c[np.mod(t,2)], **kwargs)
            t+=1
        if draw_now:
            plt.show()
        return axes

def test():
    M = 4
    glst = PixelGridList(M=M, N=16, span=np.pi/2/M*0.9)
    glst.pprint()
    glst.show2D(boundary_on=False, sc_size=10)

if __name__ == '__main__':
    test()
