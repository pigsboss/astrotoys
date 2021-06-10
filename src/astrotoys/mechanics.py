#!/usr/bin/env python
"""Celestial mechanics related toys.
"""

import numpy as np
import numexpr as ne
from scipy.optimize import minimize_scalar, minimize, basinhopping, shgo, dual_annealing
import pymath.quaternion as quaternion
from multiprocessing import Pool

AU_to_km = 149597871.0
# standard gravitational_parameters of selected bodies, in m^3/s^2.
mu_SI = {
    'sun'     : 1.32712440018e20,
    'mercury' : 2.2032e13,
    'venus'   : 3.24859e14,
    'earth'   : 3.986004418e14,
    'moon'    : 4.9048695e12,
    'mars'    : 4.282837e13,
    'ceres'   : 6.26325e10,
    'jupiter' : 1.26686534e17,
    'saturn'  : 3.7931187e16,
    'uranus'  : 5.793939e15,
    'neptune' : 6.836529e15,
    'pluto'   : 8.71e11,
    'eris'    : 1.108e12
}
DEPS = np.finfo('float64').eps
class Orbit(object):
    def __init__(self, a, ecc, inc, Ome, ome, nu0, t0=0., mu=1.):
        """Construct an elliptical orbit by elements.
a   - semi-major axis
ecc - eccentricity
inc - inclination, in rad
Ome - longitude of ascending node, in rad
ome - argument of periapsis, in rad
nu0 - true anomaly at epoch (default: J2000), in rad
t0  - epoch since J2000 (default: 0)
mu  - standard gravitational parameter (default: 1.)
      Examples:
      1. mu is in unit of G * solar_mass and the semi-major axis
         is in unit of AU, time is in unit of Jyr.
      2. mu is in unit of G * earth_mass and the semi-major axis
         is in unit GSO radius, time is in unit of sidereal day.
"""
        self.a   = a
        self.ecc = ecc
        self.inc = inc
        self.Ome = Ome
        self.ome = ome
        self.nu0 = nu0
        self.t0  = t0
        self.mu  = mu
        self.M0  = true_anomaly_to_mean_anomaly(self.nu0, self.ecc)
    def pprint(self):
        """Print with pretty format.
"""
        print("semi-major axis             : {:15.6E}".format(self.a))
        print("eccentricity                : {:11.6f}".format(self.ecc))
        print("inclination                 : {:11.6f} deg".format(np.rad2deg(self.inc)))
        print("longitude of ascending node : {:11.6f} deg".format(np.rad2deg(self.Ome)))
        print("argument of periapsis       : {:11.6f} deg".format(np.rad2deg(self.ome)))
        print("true anomaly at epoch       : {:11.6f} deg".format(np.rad2deg(self.nu0)))
        print("epoch                       : {:11.6f}".format(self.t0))
    def state(self, nu):
        """Return state vector on given true anomaly.
"""
        a   = self.a
        ecc = self.ecc
        inc = self.inc
        Ome = self.Ome
        ome = self.ome
        p       = ne.evaluate('abs(1.-ecc**2.)*a'   )
        sqrtp   = ne.evaluate('sqrt(p)'             )
        cosnu   = ne.evaluate('cos(nu)'             )
        sinnu   = ne.evaluate('sin(nu)'             )
        rho     = ne.evaluate('p/(1.+ecc*cosnu)'    )
        rx      = ne.evaluate('rho*cosnu'           )
        ry      = ne.evaluate('rho*sinnu'           )
        vr      = ne.evaluate('ecc*sinnu/sqrtp'     )
        vt      = ne.evaluate('(1.+ecc*cosnu)/sqrtp')
        vx      = ne.evaluate('vr*cosnu-vt*sinnu'   )
        vy      = ne.evaluate('vr*sinnu+vt*cosnu'   )
        x       = ne.evaluate('.5*inc'              )
        y       = ne.evaluate('.5*Ome + .5*ome'     )
        z       = ne.evaluate('.5*Ome - .5*ome'     )
        sin2x   = ne.evaluate('sin(x)*sin(x)'       )
        cos2x   = ne.evaluate('1.-sin2x'            )
        sin2y   = ne.evaluate('sin(y)*sin(y)'       )
        sin2z   = ne.evaluate('sin(z)*sin(z)'       )
        sincosy = ne.evaluate('sin(y)*cos(y)'       )
        sincosz = ne.evaluate('sin(z)*cos(z)'       )
        mxx     = ne.evaluate('-2.*sin2x*sin2z-2.*cos2x*sin2y+1.'     )
        mxy     = ne.evaluate( '2.*sin2x*sincosz-2.*cos2x*sincosy'    )
        mxz     = ne.evaluate('.5*cos(Ome-inc)-.5*cos(Ome+inc)'       )
        myx     = ne.evaluate( '2.*sin2x*sincosz+2.*cos2x*sincosy'    )
        myy     = ne.evaluate('-2.*sin2x*(1.-sin2z)-2.*cos2x*sin2y+1.')
        myz     = ne.evaluate('.5*sin(Ome-inc)-.5*sin(Ome+inc)'       )
        mzx     = ne.evaluate('.5*cos(inc-ome)-.5*cos(inc+ome)'       )
        mzy     = ne.evaluate('.5*sin(inc-ome)+.5*sin(inc+ome)'       )
        mzz     = ne.evaluate(   'cos(inc)'                           )
        r       = np.double([
            ne.evaluate('mxx*rx+mxy*ry'),
            ne.evaluate('myx*rx+myy*ry'),
            ne.evaluate('mzx*rx+mzy*ry')
        ])
        v       = np.double([
            ne.evaluate('mxx*vx+mxy*vy'),
            ne.evaluate('myx*vx+myy*vy'),
            ne.evaluate('mzx*vx+mzy*vy')
        ])
        return r, v
    def allstates(self, N=100):
        """Return equi-angular-distant state vectors (r, v)
from nu=0 to nu=2*PI on the orbit.
"""
        nu = np.arange(N)/(N-1.)*2.*np.pi
        return self.state(nu)
    def state_when(self, t):
        """Return state vector(s) at given time.
"""
        M  = self.M0 + np.sqrt(self.mu/self.a**3.)*(t-self.t0)
        nu = true_anomaly(eccentric_anomaly(M, self.ecc), self.ecc)
        return self.state(nu)
class PlanetOrbit(Orbit):
    def __init__(self, a, ecc, inc, Ome, ome, nu0, t0):
        """Construct an orbit for a planet-like object of solar system,
such as planets, asteroids as well as comets.

Arguments:
a   - semi-major axis, in AU
ecc - eccentricity
inc - inclination, in rad
Ome - ecliptical longitude of ascending node, in rad
ome - argument of perihelion, in rad
nu0 - true anomaly at epoch, in rad
t0  - epoch, in JD
"""
        super(PlanetOrbit, self).__init__(a, ecc, inc, Ome, ome, nu0, t0=t0, mu=1.)
    def pprint(self):
        """Print with pretty format.
"""
        print("semi-major axis             : {:15.6E} AU".format(self.a))
        print("eccentricity                : {:11.6f}".format(self.ecc))
        print("inclination                 : {:11.6f} deg".format(np.rad2deg(self.inc)))
        print("longitude of ascending node : {:11.6f} deg".format(np.rad2deg(self.Ome)))
        print("argument of periapsis       : {:11.6f} deg".format(np.rad2deg(self.ome)))
        print("true anomaly at epoch       : {:11.6f} deg".format(np.rad2deg(self.nu0)))
        print("epoch                       : {:11.6f} JD".format(self.t0))
    def state_when(self, t):
        """Return state vector(s) at given time, in JD.
"""
        M  = self.M0 + np.sqrt(self.mu/self.a**3.)*(t-self.t0)/365.25
        nu = true_anomaly(eccentric_anomaly(M, self.ecc), self.ecc)
        return self.state(nu)
class SatelliteOrbit(Orbit):
    def __init__(self, a, ecc, inc, Ome, ome, nu0, t0):
        """Construct an orbit for an Earth-orbit satellite.

Arguments:
a   - semi-major axis, in km
ecc - eccentricity
inc - inclination, in rad
Ome - longitude of ascending node, in rad
ome - argument of perigee, in rad
nu0 - true anomaly at epoch, in rad
t0  - epoch, in JD
"""
        super(SatelliteOrbit, self).__init__(a, ecc, inc, Ome, ome, nu0, t0=t0, mu=mu_SI['earth'])
    def pprint(self):
        """Print with pretty format.
"""
        print("semi-major axis             : {:15.6E} km".format(self.a))
        print("eccentricity                : {:11.6f}".format(self.ecc))
        print("inclination                 : {:11.6f} deg".format(np.rad2deg(self.inc)))
        print("longitude of ascending node : {:11.6f} deg".format(np.rad2deg(self.Ome)))
        print("argument of periapsis       : {:11.6f} deg".format(np.rad2deg(self.ome)))
        print("true anomaly at epoch       : {:11.6f} deg".format(np.rad2deg(self.nu0)))
        print("epoch                       : {:11.6f} JD".format(self.t0))
    def state_when(self, t):
        """Return state vector(s) at given time since epoch, in second.
"""
        M  = self.M0 + np.sqrt(self.mu/self.a**3.)*t
        nu = true_anomaly(eccentric_anomaly(M, self.ecc), self.ecc)
        return self.state(nu)
class Trajectory(Orbit):
    def __init__(self, a, ecc, inc, Ome, ome, nu, nu_start=0., nu_stop=2.*np.pi):
        """Construct an elliptical trajectory by elements and extra parameters.
Standard orbital elements:
a   - semi-major axis
ecc - eccentricity
inc - inclination
Ome - longitude of ascending node
ome - argument of periapsis
nu  - current true anomaly
Extra parameters:
nu_start    - true anomaly at start of the trajectory
nu_stop     - true anomaly at stop of the trajectory
nu_midpoint - true anomaly at midpoint of the trajectory
"""
        super(Trajectory, self).__init__(a, ecc, inc, Ome, ome, nu)
        self.nu_start    = np.mod(nu_start,    2.*np.pi)
        self.nu_stop     = np.mod(nu_stop,     2.*np.pi)
        self.nu_midpoint = np.mod(nu_midpoint, 2.*np.pi)
        if self.nu_stop > self.nu_start:
            if self.nu_midpoint > self.nu_start and self.nu_midpoint < self.nu_stop:
                self.prograde = True
            else:
                self.prograde = False
        else:
            if self.nu_midpoint < self.nu_start and self.nu_midpoint > self.nu_stop:
                self.prograde = False
            else:
                self.prograde = True
    def pprint(self):
        super(Trajectory, self).pprint()
        print("from {:f} deg to {:f} deg, via {:f} deg".format(
            np.rad2deg(self.nu_start),
            np.rad2deg(self.nu_stop),
            np.rad2deg(self.nu_midpoint)
        ))
    def allstates(self, N=100):
        """Return equi-angular-distant state vectors from nu_start to nu_stop.
"""
        if self.prograde:
            if self.nu_stop > self.nu_start:
                nu = np.arange(N)/(N-1.)*(self.nu_stop-self.nu_start)+self.nu_start
            else:
                nu = np.arange(N)/(N-1.)*(self.nu_stop+2.*np.pi-self.nu_start)+self.nu_start
        else:
            if self.nu_stop < self.nu_start:
                nu = np.arange(N)/(N-1.)*(self.nu_stop-self.nu_start)+self.nu_start
            else:
                nu = np.arange(N)/(N-1.)*(self.nu_stop-2.*np.pi-self.nu_start)+self.nu_start
        r, v = self.state(nu)
        return r, v
class RocketTrajectory(Trajectory):
    def __init__(self, a, ecc, inc, Ome, ome, nu):
        pass
def find_trajectory_mindv(orb1, orb2, mu, N=10):
    """Find the trajectory between two orbits with minimum transfer delta-v.
Global optimization is used, which could be considerably slow.

orb1 - Orbit object of the initial orbit
orb2 - Orbit object of the target orbit
mu   - standard gravitational parameter
N    - resolution of initial sampling grid

Returns:
dv   - sampled delta-v (N, N)
res  - optimization result
"""
    def eval_dv(nu):
        r1, v1 = orb1.state(nu[0])
        r2, v2 = orb2.state(nu[1])
        traj, opts = find_trajectory_mindv_s2s(r1, v1, r2, v2, mu)
        return opts['delta_v']
    if N>0:
        dv = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                dv[i,j] = eval_dv((2.*np.pi*i/N, 2.*np.pi*j/N))
        ii = np.argmin(np.min(dv, axis=1))
        jj = np.argmin(np.min(dv, axis=0))
        nu0 = (2.*np.pi*ii/N, 2.*np.pi*jj/N)
    # res = basinhopping(eval_dv, nu0, stepsize=np.pi, minimizer_kwargs={
    #     'bounds': ((0., 2.*np.pi), (0., 2.*np.pi))
    # })
    # res = dual_annealing(eval_dv, ((0., 2.*np.pi), (0., 2.*np.pi)), maxiter=100)
    res = shgo(eval_dv, ((0., 2.*np.pi), (0., 2.*np.pi)))
    # res = minimize(eval_dv, nu0, bounds=((0., 2.*np.pi), (0., 2.*np.pi)))
    return dv, res

def find_trajectory_mindv_s2s(r1, v1, r2, v2, mu, N=20, verbose=False):
    """Find the trajectory between orbit states (r1, v1) and (r2, v2) on the
same plane with minimum overall delta_v.

delta_v_overall = delta_v_entry + delta_v_exit,
where delta_v_entry is the impulse per unit of spacecraft mass that is needed
to enter the trajectory from state (r1, v1), delta_v_exit is the impulse per
unit of spacecraft mass needed to exit the trajectory and reach state (r2, v2).

The main focus is at the origin of the frame of reference.
(r1, v1) is entry orbit state vector.
(r2, v2) is exit orbit state vector.
mu is standard gravitational parameter.
N is the number of samples where delta_v is evaluated before fine optimization.
"""
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    u1 = quaternion.direction(h1)*(np.sign(h1[2])+(h1[2]==0.))
    u2 = quaternion.direction(h2)*(np.sign(h2[2])+(h2[2]==0.))
    if not np.allclose(u1, u2):
        raise ValueError('state vectors are not on the same plane.')
    r1u = quaternion.direction(r1) # unit vector along r1
    r2u = quaternion.direction(r2) # unit vector along r2
    if np.allclose(r1u, r2u):
        # degenerate to 1D oscillator
        r1mag = quaternion.norm(r1)
        r2mag = quaternion.norm(r2)
        v1p   = np.dot(r1u, v1) # magnitude of parallel component of v1
        v2p   = np.dot(r2u, v2) # magnitude of parallel component of v2
        v1t   = v1-v1p*r1u      # vertical component of v1
        v2t   = v2-v2p*r2u      # vertical component of v2
        if r1mag >= r2mag:
            if v2p >= 0.:
                dv1p = 0.
                dv2p =  np.sqrt(2.*mu*(r1mag-r2mag)/r1mag/r2mag+v1p**2.)-v2p
            else:
                dv1p = 0.
                dv2p = -np.sqrt(2.*mu*(r1mag-r2mag)/r1mag/r2mag+v1p**2.)-v2p
        else:
            if v1p >= 0.:
                dv1p =  np.sqrt(2.*mu*(r2mag-r1mag)/r1mag/r2mag+v2p**2.)-v1p
                dv2p = 0.
            else:
                dv1p = -np.sqrt(2.*mu*(r2mag-r1mag)/r1mag/r2mag+v2p**2.)-v1p
                dv2p = 0.
        dv1 = dv1p*r1u-v1t
        dv2 = v2t-dv2p*r2u
        return None, {
            'dv_entry': dv1,
            'dv_exit' : dv2,
            'delta_v' : quaternion.norm(dv1)+quaternion.norm(dv2)
        }
    def dv_all_p(t):
        _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t)
        r_entry, v_entry = orbital_motion(a, ecc, inc, Ome, ome, nu1)
        r_exit,  v_exit  = orbital_motion(a, ecc, inc, Ome, ome, nu2)
        dv_entry_p = quaternion.norm(np.double([v1[0] - v_entry[0]*np.sqrt(mu), v1[1] - v_entry[1]*np.sqrt(mu), v1[2] - v_entry[2]*np.sqrt(mu)]))
        dv_exit_p  = quaternion.norm(np.double([v2[0] - v_exit[0]*np.sqrt(mu),  v2[1] - v_exit[1]*np.sqrt(mu),  v2[2] - v_exit[2]*np.sqrt(mu)]))
        return dv_entry_p + dv_exit_p
    def dv_all_r(t):
        _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t)
        r_entry, v_entry = orbital_motion(a, ecc, inc, Ome, ome, nu1)
        r_exit,  v_exit  = orbital_motion(a, ecc, inc, Ome, ome, nu2)
        dv_entry_r = quaternion.norm(np.double([v1[0] + v_entry[0]*np.sqrt(mu), v1[1] + v_entry[1]*np.sqrt(mu), v1[2] + v_entry[2]*np.sqrt(mu)]))
        dv_exit_r  = quaternion.norm(np.double([v2[0] + v_exit[0]*np.sqrt(mu),  v2[1] + v_exit[1]*np.sqrt(mu),  v2[2] + v_exit[2]*np.sqrt(mu)]))
        return dv_entry_r + dv_exit_r
    t = ((np.arange(N)+.5)/N - .5)*np.pi
    dv_p = dv_all_p(t)
    dv_r = dv_all_r(t)
    minidx_p = np.argmin(dv_p[1:-1])
    minval_p = dv_p[minidx_p+1]
    minidx_r = np.argmin(dv_r[1:-1])
    minval_r = dv_r[minidx_r+1]
    if dv_p[minidx_p+1] < dv_p[0] and dv_p[minidx_p+1] < dv_p[-1]:
        bracket_p = (t[0], t[minidx_p+1], t[-1])
    else:
        if verbose:
            print('Convex region to minimize prograding delta-v is not found.')
            print('{:f} and {:f} are provided as starting points.'.format(t[0], t[-1]))
        bracket_p = (t[0], t[-1])
    if dv_r[minidx_r+1] < dv_r[0] and dv_r[minidx_r+1] < dv_r[-1]:
        bracket_r = (t[0], t[minidx_r+1], t[-1])
    else:
        if verbose:
            print('Convex region to minimize retrograding delta-v is not found.')
            print('{:f} and {:f} are provided as starting points.'.format(t[0], t[-1]))
        bracket_r = (t[0], t[-1])
    res_p = minimize_scalar(dv_all_p, method='brent', bracket=bracket_p)
    if res_p.success:
        t_min_p  = res_p.x
        dv_min_p = dv_all_p(t_min_p)
    else:
        if verbose:
            print('Fine optimization of prograding delta-v failed.')
        t_min_p  = t[minidx_p+1]
        dv_min_p = dv_p[minidx_p+1]
    res_r = minimize_scalar(dv_all_r, method='brent', bracket=bracket_r)
    if res_r.success:
        t_min_r  = res_r.x
        dv_min_r = dv_all_r(t_min_r)
    else:
        if verbose:
            print('Fine optimization of retrograding delta-v failed.')
        t_min_r  = t[minidx_r+1]
        dv_min_r = dv_r[minidx_r+1]
    if dv_min_r < dv_min_p:
        dv_min = dv_min_r
        t_min  = t_min_r
        _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t_min)
        r, v = orbital_motion(a, ecc, inc, Ome, ome, nu1)
        dv1  = -v-v1
        r, v = orbital_motion(a, ecc, inc, Ome, ome, nu2)
        dv2  = v2+v
        if nu2 > nu1:
            nu_mid = np.mod(.5*(nu1+nu2)+np.pi, 2.*np.pi)
        else:
            nu_mid = .5*(nu1+nu2)
    else:
        dv_min = dv_min_p
        t_min  = t_min_p
        _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t_min)
        r, v = orbital_motion(a, ecc, inc, Ome, ome, nu1)
        dv1 = v-v1
        r, v = orbital_motion(a, ecc, inc, Ome, ome, nu2)
        dv2 = v2-v
        if nu2 > nu1:
            nu_mid = .5*(nu1+nu2)
        else:
            nu_mid = np.mod(.5*(nu1+nu2)+np.pi, 2.*np.pi)
    traj = Trajectory(a, ecc, inc, Ome, ome, nu1, nu1, nu2, nu_mid)
    opts = {
        'dv_entry'           : dv1,
        'dv_exit'            : dv2,
        'delta_v'            : dv_min,
        't_opt'              : t_min,
        'nodes_prograde'     : (t, dv_p),
        'nodes_retrograde'   : (t, dv_r)
    }
    return traj, opts

def find_trajectory_byeta_s2s(r1, v1, r2, v2, mu, eta, N=1000, prograde=True):
    """Find the trajectory between orbit states (r1, v1) and (r2, v2)
with proper ETA (estimated time on arrival), in seconds.
(r1, v1) is entry orbit state vector.
(r2, v2) is exit orbit state vector.
mu is standard gravitational parameter.
eta is estimated time on arrival, in seconds.
N is the number of samples where ETA is evaluated before the optimization.
prograde indicates the direction of the trajectory from r1 to r2.
"""
    def eval_eta(t):
        _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t)
        M1 = true_anomaly_to_mean_anomaly(nu1, ecc)
        M2 = true_anomaly_to_mean_anomaly(nu2, ecc)
        return np.mod(np.sign(prograde-.5)*(M2-M1), 2.*np.pi)*np.sqrt(mu/a**3.)
    t = ((np.arange(N)+.5)/N - .5)*np.pi
    etas = eval_eta(t)
    dt2 = (etas-eta)**2.
    minidx = np.argmin(dt2[1:-1])
    minval = dt2[minidx+1]
    if minval<dt2[0] and minval<dt2[-1]:
        bracket = (t[0], t[minidx+1], t[-1])
    else:
        print('Convex region to optimize ETA is not found.')
        print('{:f} and {:f} are provided as starting points.'.format(t[0], t[-1]))
        bracket = (t[0], t[-1])
    res = minimize_scalar(lambda t:(eval_eta(t)-eta)**2., method='brent', bracket=bracket)
    if res.success:
        t_opt = res.x
    else:
        print('Fine optimization failed.')
        t_opt = t[minidx+1]
    eta_opt = eval_eta(t_opt)
    _, a, ecc, inc, Ome, ome, nu1, nu2 = find_orbits_f2p(np.double([0., 0., 0.]), r1, r2, t=t_opt)
    nu_mid = np.mod((nu1+nu2+np.pi+np.sign(prograde-.5)*np.sign((nu2>nu1)-.5)*np.pi)/2., 2.*np.pi)
    traj = Trajectory(a, ecc, inc, Ome, ome, nu1, nu1, nu2, nu_mid)
    opts = {
        'eta'   : eta_opt,
        't_opt' : t_opt,
        'nodes' : (t, etas)
    }
    return traj, opts

def find_orbits_f2p(f, r1, r2, u=None, t=None, t_rng=.95, N=100):
    """Find all possible elliptical orbits, given the main focus f as well as
two orbital position vectors r1 and r2.

f, r1 and r2 are specified with their cartesian coordinates in a fiducial 3D
frame of reference.

Elements of retrieved orbits are considered as functions an angular parameter t,
which goes from -np.pi/2 to np.pi/2. These functions diverge at t=-np.pi/2 and
t=np.pi/2. t_rng is a percentage between (-np.pi/2, np.pi/2) where the orbital
elements are calculated with N equidistant nodes, so that orbit parameterized
with an arbitrary t can later be interpolated.

Returns:
t   - angular parameter of retrieved orbital elements
a   - semi-major axes of retrieved orbits
ecc - eccentricities of retrieved orbits
"""
    if np.allclose(quaternion.direction(r1-f), quaternion.direction(r2-f)):
        raise ValueError('singularity encountered.')
    # Find the auxiliary frame of reference, where the origin lies between r1 and r2
    # and r1 lies on the positive half of the x-axis.
    o  = (r1+r2)/2.
    # Since neither entry nor exit velocity vector is available, 
    # z-axis of the aux frame is determined so that angle between entry position vector
    # r1 and exit position vector r2 is less than np.pi, as seen from the main focus,
    # which implies the orbit momentum vector h is along u.
    # if f, r1, and r2 are on the same line, z-axis of the aux frame should be specified,
    # otherwise z-axis of fiducial frame is used.
    if u is None:
        u = np.cross(r1-f, r2-f)
        if np.isclose(quaternion.norm(u), 0.):
            u = np.double([0., 0., 1.])
        else:
            u = quaternion.direction(u)
        u = u*(np.sign(u[2])+(u[2]==0.))
    a  = quaternion.direction(r1-r2) # axis-vector, i.e., x-axis of the orbit-fixed frame
    q  = quaternion.from_axes(a, u)  # quaternion that rotates the fiducial frame to the orbit-fixed frame
    qc = quaternion.conjugate(q)     # quaternion that convert coordinates from the fiducial frame to the orbit-fixed frame
    R1 = quaternion.rotate(qc, r1-o)
    R2 = quaternion.rotate(qc, r2-o)
    F1 = quaternion.rotate(qc,  f-o)
    # Let F2 be the other focus of a possible trajectory.
    # We have |R1-F1| + |R1-F2| = |R2-F1| + |R2-F2| ==>
    # |F2-R1| - |F2-R2| = |R2-F1| - |R1-F1|.
    # Find the hyperbola with R1 and R2 as the two foci and a_h = ||R1-F1| - |R2-F1|| / 2 as the semi-major axis,
    # so that the given focus, i.e., the origin of the orbit-fixed frame is on one branch of the hyperbola,
    # and all possible positions of the other focus are on the other branch of the hyperbola.
    a_h = np.abs(quaternion.norm(R1-F1) - quaternion.norm(R2-F1)) / 2.
    c_h = quaternion.norm(R1-R2) / 2.
    b_h = (c_h**2. - a_h**2.)**.5
    if t is None:
        t   = ((np.arange(0,N)+.5)/N - .5)*t_rng*np.pi
    if np.isclose(F1[0], 0.):
        # the hyperpola degenerates into the middle perpendicular of R1-R2, i.e., the y-axis of the aux frame.
        x = np.zeros_like(t)
    elif F1[0] > 0.:
        # the given focus lies on the right branch of the hyperbola
        x = a_h/np.cos(t+np.pi)
    else:
        # the given focus lies on the left branch of the hyperbola
        x = a_h/np.cos(t)
    y   = b_h*np.tan(t)
    F2  = np.double([x, y, np.zeros_like(x)])
    c   = .5*((F1[0]-x)**2. + (F1[1]-y)**2.)**.5
    a   = .5*(((R1[0]-F1[0])**2.+(R1[1]-F1[1])**2.)**.5+((R1[0]-x)**2.+(R1[1]-y)**2.)**.5)
    ecc = c / a
    # inclination
    inc = np.arccos(u[2])
    # longitude of ascending node
    if np.isclose(np.abs(u[2]), 1.):
        Ome = 0.
    else:
        n   = quaternion.direction(np.double([-u[1], u[0], 0.])) # node vector
        Ome = np.mod(np.arccos(n[0])*(np.sign(n[1])+(u[1]==0)), 2.*np.pi)
    try:
        # eccentricity vectors: pointing from apoapsis to periapsis, with magnitude equal to the eccentricy.
        ome    = np.empty_like(ecc)
        iscirc = np.isclose(ecc, 0.)
        f2     = quaternion.rotate(q, np.double([x, y, np.zeros_like(x)])) + o.reshape((3, 1))
        ehat   = ecc * quaternion.direction(f.reshape((3, 1)) - f2)
        # arguments of periapsis
        if np.isclose(np.abs(u[2]), 1.):
            ome[iscirc]  = 0.
            ome[~iscirc] = np.mod(np.arctan2(ehat[1][~iscirc], ehat[0][~iscirc])*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
        else:
            ome[iscirc]  = 0.
            ome[~iscirc] = np.mod(
                np.arccos((
                    n[0]*ehat[0][~iscirc]+
                    n[1]*ehat[1][~iscirc]+
                    n[2]*ehat[2][~iscirc]
                )/ecc[~iscirc])*(np.sign(ehat[2][~iscirc])+(ehat[2]==0)),
                2.*np.pi)
        # true anomalies
        f1r1 = R1 - F1
        f2f1 = F1.reshape((3,1)) - F2
        f1r2 = R2 - F1
        nu1  = np.mod((np.arctan2(f1r1[1], f1r1[0]) - np.arctan2(f2f1[1], f2f1[0]))*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
        nu2  = np.mod((np.arctan2(f1r2[1], f1r2[0]) - np.arctan2(f2f1[1], f2f1[0]))*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
        # nu1  = np.arctan2(f1r1[1], f1r1[0]) - np.arctan2(f2f1[1], f2f1[0])
        # nu2  = np.arctan2(f1r2[1], f1r2[0]) - np.arctan2(f2f1[1], f2f1[0])
    except TypeError:
        f2     = quaternion.rotate(q, np.double([x, y, 0.])) + o
        ehat   = ecc * quaternion.direction(f - f2)
        # arguments of periapsis
        if np.isclose(np.abs(u[2]), 1.):
            if np.isclose(ecc, 0.):
                ome = 0.
            else:
                ome = np.mod(np.arctan2(ehat[1], ehat[0])*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
        else:
            if np.isclose(ecc, 0.):
                ome = 0.
            else:
                ome = np.mod(np.arccos((n[0]*ehat[0]+n[1]*ehat[1]+n[2]*ehat[2])/ecc)*(np.sign(ehat[2])+(ehat[2]==0)), 2.*np.pi)
        # true anomalies
        f1r1 = R1 - F1
        f2f1 = F1 - F2
        f1r2 = R2 - F1
        nu1  = np.mod((np.arctan2(f1r1[1], f1r1[0]) - np.arctan2(f2f1[1], f2f1[0]))*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
        nu2  = np.mod((np.arctan2(f1r2[1], f1r2[0]) - np.arctan2(f2f1[1], f2f1[0]))*(np.sign(u[2])+(u[2]==0)), 2.*np.pi)
    return t, a, ecc, inc, Ome, ome, nu1, nu2
    
def find_orbit(r, v, mu):
    """Find classical orbital elements from given position and velocity.
r  - position
v  - velocity
mu - standard gravitational parameter, i.e., G*(m1+m2)

Returns:
a   - semi-major axis
ecc - eccentricity
inc - inclination, in rad
Ome - longitude of ascending node, in rad
ome - argument of periapsis, in rad
nu  - true anomaly, in rad
"""
    h = np.array([r[1]*v[2]-r[2]*v[1],r[2]*v[0]-r[0]*v[2],r[0]*v[1]-r[1]*v[0]])
    hmag = quaternion.norm(h)
    nhat = np.array([-h[1], h[0], np.zeros_like(h[2])])
    nmag = quaternion.norm(nhat)
    v2   = quaternion.norm(v)**2.
    rmag = quaternion.norm(r)
    ehat = ((v2-mu/rmag)*r - np.sum(r*v, axis=0)*v)/mu
    ecc  = quaternion.norm(ehat)
    w    = v2/2. - mu/rmag
    try:
        para = np.isclose(ecc, 1.)
        a    = np.empty_like(ecc)
        p    = np.empty_like(ecc)
        a[para]  = np.inf
        a[~para] = -mu[~para]/2./w[~para]
        p[para]  = hmag[para]**2./mu[para]
        p[~para] = a[~para]*(1.-ecc[~para]**2.)
        inc  = np.arccos(h[2]/hmag)
        equa = np.isclose(np.abs(h[2]), hmag)
        Ome  = np.empty_like(ecc)
        ome  = np.empty_like(ecc)
        nu   = np.empty_like(ecc)
        Ome[equa]  = 0.
        Ome[~equa] = np.arccos(nhat[0,~equa]/nmag[~equa])
        Ome[~equa] = np.mod(Ome[~equa]*(np.sign(nhat[1,~equa])+(nhat[1,~equa]==0)), 2.*np.pi)
        circ = np.isclose(ecc, 0.)
        ome[circ]  = 0.
        ome[(~circ)&equa] = np.arctan2(ehat[1,(~circ)&equa], ehat[0,(~circ)&equa])
        ome[(~circ)&equa] = np.mod(ome[(~circ)&equa]*(np.sign(h[2,(~circ)&equa])+(h[2,(~circ)&equa]==0)), 2.*np.pi)
        ome[~(circ|equa)] = np.arccos(np.sum(nhat[:,~(circ|equa)]*ehat[:,~(circ|equa)], axis=0)/
                                      nmag[~(circ|equa)]/ecc[~(circ|equa)])
        ome[~(circ|equa)] = np.mod(ome[~(circ|equa)]*(np.sign(ehat[2,~(circ|equa)])+(ehat[2,~(circ|equa)]==0)), 2.*np.pi)
        nu[circ&equa]     = np.arccos(r[0,circ&equa]/rmag[circ&equa])
        nu[circ&equa]     = np.mod(nu[circ&equa]*(np.sign(-v[0,circ&equa])+(v[0,circ&equa]==0)), 2.*np.pi)
        nu[circ&(~equa)]  = np.arccos(np.sum(nhat[:,circ&(~equa)]*r[:,circ&(~equa)], axis=0)/
                                      nmag[circ&(~equa)]/rmag[circ&(~equa)])
        nu[circ&(~equa)]  = np.mod(nu[circ&(~equa)]*(np.sign(r[2,circ&(~equa)])+(r[2,circ&(~equa)]==0)), 2.*np.pi)
        nu[~circ]  = np.arccos(np.sum(ehat[:,~circ]*r[:,~circ], axis=0)/ecc[~circ]/rmag[~circ])
        rdv = np.sum(r[:,~circ]*v[:,~circ], axis=0)
        nu[~circ]  = np.mod(nu[~circ]*(np.sign(rdv)+(rdv==0)), 2.*np.pi)
    except TypeError:
        if np.isclose(ecc, 1.): # parabolic trajectory
            a = np.inf
            p = hmag**2./mu
        else:                   # elliptical or hyperbolic trajectory
            a = -mu/2./w
            p = a*(1.-ecc**2.)
        inc = np.arccos(h[2]/hmag)
        if np.isclose(np.abs(h[2]), hmag): # equatorial orbit
            Ome = 0.
        else:
            Ome = np.arccos(nhat[0]/nmag)
            if nhat[1]<0.:
                Ome = 2.*np.pi-Ome
        if np.isclose(ecc, 0.): # circular orbit
            ome = 0.
        else:
            if np.isclose(np.abs(h[2]), hmag):
                ome = np.arctan2(ehat[1], ehat[0])
                if h[2]<0.:
                    ome = 2.*np.pi-ome
            else:
                ome = np.arccos(np.dot(nhat, ehat)/nmag/ecc)
                if ehat[2]<0.:
                    ome = 2.*np.pi-ome
        if np.isclose(ecc, 0.):
            if np.isclose(np.abs(h[2]), hmag):
                nu = np.arccos(r[0]/rmag)
                if v[0]>0.:
                    nu = 2.*np.pi-nu
            else:
                nu = np.arccos(np.dot(nhat, r)/nmag/rmag)
                if r[2]<0.:
                    nu = 2.*np.pi-nu
        else:
            nu = np.arccos(np.dot(ehat, r)/ecc/rmag)
            if np.dot(r,v)<0.:
                nu = 2.*np.pi-nu
    return a, ecc, inc, Ome, ome, nu

def orbital_motion(a, ecc, inc, Ome, ome, nu):
    """Find motion parameters (position and velocity) of an object on an elliptical orbit,
given the orbital elements.
a   - semi-major axis
ecc - eccentricity
inc - inclination, in rad
Ome - longitude of ascending node, in rad
ome - argument of periapsis, in rad
nu  - true anomaly, in rad

Returns:
r - position vector, in the same dimension as a.
v - normalized velocity vector, in UNIT_OF_LENGTH / UNIT_OF_TIME / sqrt(standard gravitational parameter).
"""
    p = np.abs(1. - ecc**2.) * a # semi-latus, valid for circular, elliptical and hyperbolic orbits. 
    q = qo2e(Ome, inc, ome) # quaternion that converts coordinates on orbital plane to initial coordinate system.
    rho = p / (1. + ecc*np.cos(nu))
    r = np.array([rho*np.cos(nu), rho*np.sin(nu), np.zeros_like(rho)])
    v_r = ecc*np.sin(nu)/np.sqrt(p)
    v_t = (1.+ecc*np.cos(nu))/np.sqrt(p)
    v = np.array([v_r*np.cos(nu) - v_t*np.sin(nu), v_r*np.sin(nu) + v_t*np.cos(nu), np.zeros_like(rho)])
    return quaternion.rotate(q, r), quaternion.rotate(q, v)

def orbital_elements_to_rotation_matrix_inverse(Ome, inc, ome):
    return np.matrix([
        [-np.sin(Ome)*np.cos(inc)*np.sin(ome) + np.cos(Ome)*np.cos(ome),  np.cos(Ome)*np.cos(inc)*np.sin(ome) + np.sin(Ome)*np.cos(ome), np.sin(inc)*np.sin(ome)],
        [-np.sin(Ome)*np.cos(inc)*np.cos(ome) - np.cos(Ome)*np.sin(ome),  np.cos(Ome)*np.cos(inc)*np.cos(ome) - np.sin(Ome)*np.sin(ome), np.sin(inc)*np.cos(ome)],
        [                              np.sin(Ome)*np.sin(inc),                              -np.cos(Ome)*np.sin(inc),          np.cos(inc)]])

def orbital_elements_to_rotation_matrix(Ome, inc, ome):
    return np.matrix([
        [-np.sin(Ome)*np.sin(ome)*np.cos(inc) + np.cos(Ome)*np.cos(ome), -np.sin(Ome)*np.cos(inc)*np.cos(ome) - np.sin(ome)*np.cos(Ome),  np.sin(Ome)*np.sin(inc)],
        [ np.sin(Ome)*np.cos(ome) + np.sin(ome)*np.cos(Ome)*np.cos(inc), -np.sin(Ome)*np.sin(ome) + np.cos(Ome)*np.cos(inc)*np.cos(ome), -np.sin(inc)*np.cos(Ome)],
        [                              np.sin(inc)*np.sin(ome),                               np.sin(inc)*np.cos(ome),           np.cos(inc)]])
    
def find_moid(orbital_elements_a, orbital_elements_b):
    ## http://acta.astrouw.edu.pl/Vol63/n2/pdf/pap_63_2_10.pdf
    a_a, e_a, Ome_a, inc_a, ome_a = orbital_elements_a
    a_b, e_b, Ome_b, inc_b, ome_b = orbital_elements_b
    orbmat = (orbital_elements_to_rotation_matrix(Ome_a, inc_a, ome_a)**-1) * orbital_elements_to_rotation_matrix(Ome_b, inc_b, ome_b)
    Dp  = 0.
    Dpp = 0.
    Dm  = np.zeros((4,))
    Dm[:] = -1.
    Nu_a = np.copy(Dm)
    Nu_b = np.copy(Dm)
    nu_alr = np.zeros((3,))
    nu_blr = np.zeros((3,))
    dm_lr  = np.zeros((3,3))
    t   = 0
    for nu_b in np.arange(0, 2.*np.pi+.5, .12):
        r_b = a_b*(1-e_b**2.) / (1.+e_b*np.cos(nu_b))
        x_b, y_b, z_b = np.squeeze(np.array(orbmat * np.matrix([[r_b*np.cos(nu_b)], [r_b*np.sin(nu_b)], [0.]])))
        nu_a = np.arctan2(y_b, x_b)
        rho_b = (x_b**2. + y_b**2.)**.5
        r_a = a_a*(1-e_a**2.) / (1.+e_a*x_b/rho_b)
        D0 = (z_b**2. + (rho_b - r_a)**2.)**.5
        if Dp<Dpp and Dp<D0:
            Dm[t] = Dp
            Nu_a[t] = nu_a-.12
            Nu_b[t] = nu_b-.12
            t += 1
        Dpp = Dp
        Dp = D0
    for tt in range(t):
        dnu = 0.06
        while dnu>1e-18:
            nu_alr[0] = Nu_a[tt]-dnu
            nu_alr[1] = Nu_a[tt]
            nu_alr[2] = Nu_a[tt]+dnu
            nu_blr[0] = Nu_b[tt]-dnu
            nu_blr[1] = Nu_b[tt]
            nu_blr[2] = Nu_b[tt]+dnu
            Dp = Dm[tt]
            for i in range(3):
                r_b = a_b*(1.-e_b**2.) / (1.+e_b*np.cos(nu_blr[i]))
                x_b, y_b, z_b = np.squeeze(np.array(orbmat * np.matrix([[r_b*np.cos(nu_blr[i])], [r_b*np.sin(nu_blr[i])], [0.]])))
                for j in range(3):
                    r_a = a_a*(1.-e_a**2.) / (1.+e_a*np.cos(nu_alr[j]))
                    x_a = r_a*np.cos(nu_alr[j])
                    y_a = r_a*np.sin(nu_alr[j])
                    dm_lr[i,j] = ((x_a-x_b)**2. + (y_a-y_b)**2. + z_b**2.)**.5
                    if dm_lr[i,j] < Dm[tt]:
                        Dm[tt] = dm_lr[i,j]
                        Nu_a[tt] = nu_alr[j]
                        Nu_b[tt] = nu_blr[i]
            print(dm_lr)
            if (Dm[tt] == Dp):
                dnu = dnu*.15
    print(Dm)
    print(Nu_a)
    print(Nu_b)
    return

def orbitvtx_quat(a=1.0, ecc=0.0, nu0=0.0, Ome=0.0, dec=0.0, ome=0.0):
    """Orbit vertices

a    - semi-major axis (elliptical orbit and hyperbolic trajectory) or 
       half semi-latus rectum (parabolic trajectory).
ecc  - eccentricity.
nu0  - true anomaly at epoch, in rad.
Ome  - longitude of ascending node.
dec  - inclination (angle from ecliptic plane to orbit plane at ascending node), in rad.
ome  - argument of the periapsis, in rad.
"""
    if ecc<1:
        a  = np.abs(a)
        nu = np.arange(0, 1.001, 0.001)*np.pi*2.0
        u  = a*np.cos(nu) - a*ecc
        v  = np.sqrt(1.0 - ecc**2.0)*a*np.sin(nu)
    elif ecc>1:
        a  = -np.abs(a)
        nua = np.arccos(-1.0/ecc)
        nu  = (np.arange(0.01, 1.00, 0.01)-0.5)*2.0*nua
        rho = a*(1.0 - ecc**2.0)/(1.0 + ecc*np.cos(nu))
        u   = rho*np.cos(nu)
        v   = rho*np.sin(nu)
    else:
        v = (np.arange(0, 1.01, 0.01)-0.5)*8.0*a
        u = -((v**2.0 / 4.0*a) - a)
    return quaternion.rotate(qo2e(Ome, dec, ome), (u,v,np.zeros(u.shape)))

def orbitvtx(a=1.0, ecc=0.0, nu0=0.0, Ome=0.0, dec=0.0, ome=0.0):
    """Orbit vertices

a    - semi-major axis (elliptical orbit and hyperbolic trajectory) or 
       half semi-latus rectum (parabolic trajectory).
ecc  - eccentricity.
nu0  - true anomaly at epoch, in rad.
Ome  - longitude of ascending node.
dec  - inclination (angle from ecliptic plane to orbit plane at ascending node), in rad.
ome  - argument of the periapsis, in rad.
"""
    if ecc<1:
        a  = np.abs(a)
        nu = np.arange(0, 1.001, 0.001)*np.pi*2.0
        u  = a*np.cos(nu) - a*ecc
        v  = np.sqrt(1.0 - ecc**2.0)*a*np.sin(nu)
    elif ecc>1:
        a  = -np.abs(a)
        nua = np.arccos(-1.0/ecc)
        nu  = (np.arange(0.01, 1.00, 0.01)-0.5)*2.0*nua
        rho = a*(1.0 - ecc**2.0)/(1.0 + ecc*np.cos(nu))
        u   = rho*np.cos(nu)
        v   = rho*np.sin(nu)
    else:
        v = (np.arange(0, 1.01, 0.01)-0.5)*8.0*a
        u = -((v**2.0 / 4.0*a) - a)
    return np.array([
        u*(-np.sin(Ome)*np.sin(ome)*np.cos(dec) + np.cos(Ome)*np.cos(ome)) - v*(np.sin(Ome)*np.cos(dec)*np.cos(ome) + np.sin(ome)*np.cos(Ome)),
        u*(np.sin(Ome)*np.cos(ome) + np.sin(ome)*np.cos(Ome)*np.cos(dec)) + v*(-np.sin(Ome)*np.sin(ome) + np.cos(Ome)*np.cos(dec)*np.cos(ome)),
        u*np.sin(dec)*np.sin(ome) + v*np.sin(dec)*np.cos(ome)])

def qo2e(Ome, dec, ome):
    """Quaternion that converts coordinates on orbit plane to ecliptic plane.

Ome - longitude of the ascending node, in rad.
dec - inclination of the orbit plane, in rad.
ome - argument of periapsis, in rad.
"""
    a = ne.evaluate('.5*dec')
    b = ne.evaluate('.5*Ome + .5*ome')
    c = ne.evaluate('.5*Ome - .5*ome')
    return np.double([
        ne.evaluate('cos(a)*cos(b)'),
        ne.evaluate('sin(a)*cos(c)'),
        ne.evaluate('sin(a)*sin(c)'),
        ne.evaluate('cos(a)*sin(b)')
    ])
## def qo2e_ref(Ome, dec, ome):
##     qz = np.double([
##         np.cos(Ome*0.5),
##         np.zeros_like(Ome),
##         np.zeros_like(Ome),
##         np.sin(Ome*0.5)])
##     X  = quaternion.rotate(qz, [1.0, 0.0, 0.0])
##     qX = np.double([
##         np.cos(dec*0.5),
##         np.sin(dec*0.5)*X[0],
##         np.sin(dec*0.5)*X[1],
##         np.sin(dec*0.5)*X[2]])
##     Z  = quaternion.rotate(qX, [0.0, 0.0, 1.0])
##     qZ = np.double([
##         np.cos(ome*0.5),
##         np.sin(ome*0.5)*Z[0],
##         np.sin(ome*0.5)*Z[1],
##         np.sin(ome*0.5)*Z[2]])
##     return quaternion.multiply(qZ, quaternion.multiply(qX, qz))


def qe2o(Ome, dec, ome):
    """Quaternion that converts coordinates on ecliptic plane to orbit plane.

Ome - longitude of the ascending node, in rad.
dec - inclination of the orbit plane, in rad.
ome - argument of the periapsis, in rad.
"""
    return quaternion.conjugate(qo2e(Ome, dec, ome))

def coordinates_on_ellipse(t, t0=0, nu0=0, M0=None, T=None, a=1.0, ecc=0.0):
    """coordinates on elliptical orbit plane of solar system body at given time.

Input:
t    - current time, in Jyr (Julian years).
t0   - reference epoch, in Jyr.
nu0  - true anomaly at epoch, in rad.
M0   - mean anomaly at epoch, in rad.
T    - orbit period, in Jyr.
a    - semi-major axis (for elliptical or hyperbolic orbit), in AU.
ecc  - eccentricity.
"""
    if T is None:
        T = a**(1.5)
    if M0 is None:
        c    = a*ecc
        b    = a*(1.0 - ecc**2.0)**0.5
        rho0 = a*(1.0-ecc**2.0) / (1.0+ecc*np.cos(nu0))
        x0   = rho0*np.cos(nu0)
        y0   = rho0*np.sin(nu0)
        E0   = np.arctan2(y0/b, (x0+c)/a)
        M0   = E0 - ecc*np.sin(E0)
    M    = np.mod((t-t0)*2.0*np.pi/T + M0, 2.0*np.pi)
    E    = eccentric_anomaly(M, ecc)
    x    = a*np.cos(E)-c
    y    = b*np.sin(E)
    rho  = (x**2.0+y**2.0)**0.5
    nu   = np.arctan2(y,x)
    return x,y,rho,nu

def coordinates_on_hyperbola(t, t0=0, nu0=0, a=1.0, ecc=0.0):
    """coordinates on hyperbolic orbit plane of solar system body at given time.

Input:
t    - current time, in Jyr (Julian years).
t0   - reference epoch, in Jyr.
nu0  - true anomaly at epoch, in rad.
a    - semi-major axis (for elliptical or hyperbolic orbit), in AU.
ecc  - eccentricity.
"""
    c    = a*ecc
    b    = a*(ecc**2.0 - 1.0)**0.5
    rho0 = a*(1.0-ecc**2.0) / (1.0+ecc*np.cos(nu0))
    x0   = rho0*np.cos(nu0)
    y0   = rho0*np.sin(nu0)
    H0   = np.arctanh((y0*a)/((x0+c)*b))
    M0   = ecc*np.sinh(H0) - H0
    M    = (t-t0)*((-a)**(-1.5))*2.0*np.pi + M0
    H    = hyperbolic_anomaly(M, ecc)
    x    = a*np.cosh(H)-c
    y    = b*np.sinh(H)
    rho  = (x**2.0+y**2.0)**0.5
    nu   = np.arctan2(y,x)
    return x,y,rho,nu

def coordinates_on_parabola(t, t0=0, nu0=0, a=1.0):
    """coordinates on parabolic orbit plane of solar system body at given time.

Input:
t    - current time, in Jyr (Julian years).
t0   - reference epoch, in Jyr.
nu0  - true anomaly at epoch, in rad.
a    - half of semi-latus rectum for parabolic orbit, in AU.
"""
    M0  = (3.0*np.tan(nu0/2.0) + np.tan(nu0/2.0)**3.0)/12.0
    M   = (t-t0)*((2.0*a)**(-1.5))*np.pi + M0
    N   = (3.0*M + (9.0*M**2.0 + 1.0))**(1.0/3.0)
    nu  = 2.0*np.arctan(N-1.0/N)
    rho = 2.0*a / (1.0+np.cos(nu))
    x   = rho*np.cos(nu)
    y   = rho*np.sin(nu)
    return x,y,rho,nu

def hyperbolic_anomaly(M, ecc, max_loops=100):
    """Calculate hyperbolic anomaly from mean anomaly and eccentricity.

M is mean anomaly.
ecc is eccentricity.
max_loops is the maximum number of loops.

The hyperbolic anomaly H is calculated by solving the following equation
with fixed point iteration:
M = ecc * sinh(H) - H
"""
    H  = M
    t  = 0
    d  = np.pi
    while t < max_loops and not np.allclose(d, 0.):
        Hn = np.arcsinh((H+M)/ecc)
        d  = Hn - H
        H  = Hn
        t += 1
    return H

def eccentric_anomaly(M, ecc, max_loops=100):
    """Calculate eccentric anomaly from mean anomaly and eccentricity.

M         is mean anomaly.
ecc       is eccentricity.
max_loops is the maximum number of loops.

The eccentric anomaly E is calculated by solving the following equation
with Newton's method:
M = E - ecc * sin(E)
"""
    E = M
    t = 0
    d = np.pi
    while t < max_loops and not np.allclose(d, 0.):
        d  = (E - ecc*np.sin(E) - M) / (1.0 - ecc*np.cos(E))
        E  = E - d
        t += 1
    return E

def true_anomaly(E, ecc):
    """Calculate true anomaly from eccentric anomaly.

E   is eccentric anomaly (if ecc < 1) or hyperbolic anomaly (if ecc > 1).
ecc is eccentricity.
"""
    a = np.sign(1-ecc)
    c = a*ecc
    b = np.abs(1.0 - ecc**2.0)**0.5
    x = 0.5*(1+np.sign(1-ecc))*(a*np.cos(E)-c) + 0.5*(1-np.sign(1-ecc))*(a*np.cosh(E)-c)
    y = 0.5*(1+np.sign(1-ecc))*(b*np.sin(E))   + 0.5*(1-np.sign(1-ecc))*(b*np.sinh(E))
    return np.arctan2(y,x)

def true_anomaly_to_mean_anomaly(f, e):
    """Compute mean anomaly from eccentricity and true anomaly.
f is true anomaly, in rad.
e is eccentricity.
"""
    E = np.arctan2(np.sqrt(1. - e**2.)*np.sin(f), e + np.cos(f))
    return E - e*np.sin(E)

def mean_anomaly_to_true_anomaly_series(M, e):
    """Compute true anomaly from eccentricity and mean anomaly with series expansion.
M is mean anomaly in radian [0, 2pi).
e is eccentricity.
"""
    return M + \
        (2.*e - 0.25*e**3.)*np.sin(M) + \
        1.25*e**2.*np.sin(2.*M) + \
        13./12.*e**3.*np.sin(3.*M) + \
        103./96.*e**4.*np.sin(4.*M) + \
        1097./960.*e**5.*np.sin(5.*M) + \
        1223./960.*e**6.*np.sin(6.*M) + \
        47273./32256.*e**7.*np.sin(7.*M)

def true_anomaly_to_mean_anomaly_series(f, e):
    """Compute mean anomaly from eccentricity and true anomaly with series expansion.
f is true anomaly in radian [0, 2pi).
e is eccentricity.
"""
    return f - \
        2.*e*np.sin(f) + \
        (0.75*e**2. + 0.125*e**4.)*np.sin(2.*f) - \
        1./3.*e**3.*np.sin(3.*f) + \
        5./32.*e**4.*np.sin(4.*f)
