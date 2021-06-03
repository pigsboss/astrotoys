#!/usr/bin/env python
"""Photometry related toys.
"""

import numpy as np

h          = 6.626e-34   # Planck constant, in Js
c          = 2.998e8     # speed of light, in m/s
kB         = 1.381e-23   # Boltzmann constant, in J/K
AU         = 1.496e11    # Astromonical Unit, in m
PC         = 3.086e16    # Parsec, in m
sigma      = 5.670e-8    # Stefan-Boltzmann constant, in W/m2/K4
R_sun      = 6.957e8     # in m
T_sun      = 5.778e3     # in K
T_vega     = 9.602e3     # Vega temperature in K

def photonenergy(l):
    """Photon energy in ergs.

l  - wavelength in cm.
"""
    return 1e7*h*c/(l*0.01)
passbands  = {
    'U':{'lambda':np.arange(300.0,425.0,5.0),
         'filter':np.double([0.00,0.016,0.068,0.167,0.287,0.423,0.560,0.673,0.772,0.841,0.905,0.943,0.981,0.993,1.000,0.989,0.916,0.804,0.625,0.423,0.238,0.114,0.051,0.019,0.000])},
    'B':{'lambda':np.arange(360.0,561.0,10.0),
         'filter':np.double([0.0,0.030,0.134,0.567,0.920,0.978,1.000,0.978,0.935,0.853,0.740,0.640,0.536,0.424,0.325,0.235,0.150,0.095,0.043,0.009,0.0])},
    'V':{'lambda':np.arange(470.0,701.0,10.0),
         'filter':np.double([0.000,0.030,0.163,0.458,0.780,0.967,1.000,0.973,0.898,0.792,0.684,0.574,0.461,0.359,0.270,0.197,0.135,0.081,0.045,0.025,0.017,0.013,0.009,0.000])},
    'R':{'lambda':np.arange(550.0,901.0,10.0),
         'filter':np.double([0.000,0.23,0.74,0.91,0.98,1.000,0.98,0.96,0.93,0.90,0.86,0.81,0.78,0.72,0.67,0.61,0.56,0.51,0.46,0.40,0.35,0.14,0.03,0.00])},
    'I':{'lambda':np.arange(700.0,921.0,10.0),
         'filter':np.double([0.000,0.024,0.232,0.555,0.785,0.910,0.965,0.985,0.990,0.995,1.000,1.000,0.990,0.980,0.950,0.910,0.860,0.750,0.560,0.330,0.150,0.030,0.000])}}

EXP_MAX    = np.floor(np.log(np.finfo("double").max))

def bbl(T,l):
    """Black body radiation power (W) per effective area (m2) per solid angle (sr)
per wavelength (m).
T is temperature in K.
l is wavelength in m.
"""
    return (2.0*h*(c**2.0)/(l**5.0)) / (np.exp(np.clip((h*c)/(kB*T*l), 0.0, EXP_MAX)) - 1.0)

def phasecurve(a, G=0.15):
    """Calculate value of phase curve with given phase angle.

a is the phase angle, in radian.
G is the slope parameter of the classic H,G magnitude phase function.

Reference:
Muinonen K, et al. A three-parameter magnitude phase function for asteroids[J]. Icarus, 2010, 209(2): 542-555.
"""
    a  = np.pi-np.abs(np.pi-np.mod(a,np.pi*2.0))
    p1 = np.exp(-3.33*(np.tan(0.5*a)**0.63))
    p2 = np.exp(-1.87*(np.tan(0.5*a)**1.22))
    return (1.0-G)*p1 + G*p2

def mpd2h(d, albedo=0.3):
    """Calculate absolute magnitude of a minor planet from its diameter.

d is diameter in kilo-meters

log10(d) = 3.1236 - 0.2H - 0.5log10(p), where p is the geometric albedo.

Reference:
Muinonen K, et al. A three-parameter magnitude phase function for asteroids[J]. Icarus, 2010, 209(2): 542-555.
"""
    return 5.0*(3.1236 - np.log10(d) - 0.5*np.log10(albedo))

def mph2d(h, albedo=0.3):
    """Calculate diameter of a minor planet from its absolute magnitude.

h is absolute magnitude.

log10(d) = 3.1236 - 0.2H - 0.5log10(p), where p is the geometric albedo.

Reference:
Muinonen K, et al. A three-parameter magnitude phase function for asteroids[J]. Icarus, 2010, 209(2): 542-555.
"""
    return 10.0**(3.1236 - 0.2*h - 0.5*np.log10(albedo))

def vmag2flux(m):
    """Convert V-band magnitude to flux in Johnson-Cousins system.

Input:
m is V-band apparent magnitude

Return:
flux (power per unit area per unit wavelength), in erg/s/cm2/nm

Reference:
Flux at m=0 and lambda=550nm is 3.64e-20 erg/s/cm2/Hz
3.64e-20 erg/s/cm2/Hz * 2.998e17 nm Hz / (550 nm)**2.0 = 3.6e-8 erg/s/cm2
"""
    return 3.6 * 10.0**(-8-m/2.5)

def vmag2photrate(m):
    """Convert V-band magnitude to photon rate in Johnson-Cousins system.

Input:
m is V-band apparent magnitude

Return:
photon rate (photon number per second per unit area) in photons/s/cm2

Reference:
Flux at m=0 and lambda=550nm is 3.64e-20 erg/s/cm2/Hz
bandwidth is 88 nm
3.64e-20 erg/s/cm2/Hz * 2.998e17 nm Hz / (550 nm)**2.0 = 3.6e-8 erg/s/cm2/nm
3.6e-8 erg/s/cm2 * 88 nm * 550nm / (2.998e17 nm Hz * 6.63e-27 erg s) = 8.8e5 photons/s/cm2
"""
    return 8.8 * 10.0**(5-m/2.5)

def rmag2flux(m):
    """Convert R-band magnitude to flux in Johnson-Cousins system.

Input:
m is R-band apparent magnitude

Return:
flux (power per unit area per unit wavelength), in erg/s/cm2/nm

Reference:
Flux at m=0 and lambda=640nm is 3.08e-20 erg/s/cm2/Hz
3.08e-20 erg/s/cm2/Hz * 2.998e17 nm Hz / (640 nm)**2.0 = 2.3e-8 erg/s/cm2/nm
"""
    return 2.3 * 10.0**(-8-m/2.5)

def rmag2photrate(m):
    """Convert R-band magnitude to photon rate in Johnson-Cousins system.

Input:
m is R-band apparent magnitude

Return:
photon rate (photon number per second per unit area) in photons/s/cm2

Reference:
Flux at m=0 and lambda=640nm is 3.08e-20 erg/s/cm2/Hz
bandwidth is 145 nm
3.08e-20 erg/s/cm2/Hz * 2.998e17 nm Hz / (640 nm)**2.0 = 2.3e-8 erg/s/cm2/nm
2.3e-8 erg/s/cm2/nm * 145 nm * 640 nm / (2.998e17 nm Hz * 6.63e-27 erg s) = 1.1e6 photons/s/cm2
"""
    return 1.1 * 10.0**(6-m/2.5)

def gmag2photrate(m):
    """Convert Gaia G-band magnitude to photon rate, approximated with V-band in Johnson-Cousins system.

Input:
m is Gaia G-band apparent magnitude

Return:
photon rate (photon per unit area per second), in photons/s/cm2

Reference:
Gaia G band mean wavelength and bandwidth in Johnson-Cousins system are 673nm and 440nm.
Vega flux at 550nm (V-band) is 3.56e-8 erg/s/cm2/nm.
Vega flux at 673nm is 2.24e-8 erg/s/cm2/nm approximated with BB spectral model.
2.24e-8 erg/s/cm2/nm * 440 nm * 673e-9 m / (3e8 m/s * 6.63e-27 erg s) = 3.3e6 photons/s/cm2
(Jordi et al., 2010)
"""
    return 3.3 * 10.0**(6-m/2.5)
