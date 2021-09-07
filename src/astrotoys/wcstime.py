"""Celestial coordinate systems, terrestrial coordinate systems
and time systems, as well as transformations, for astronomy and
astrophysics.

Announced leap seconds to date (https://www.ietf.org/timezones/data/leap-seconds.list):
In datetime strings:
[   '19720701',
    '19730101',
    '19740101',
    '19750101',
    '19760101',
    '19770101',
    '19780101',
    '19790101',
    '19800101',
    '19810701',
    '19820701',
    '19830701',
    '19850701',
    '19880101',
    '19900101',
    '19910101',
    '19920701',
    '19930701',
    '19940701',
    '19960101',
    '19970701',
    '19990101',
    '20060101',
    '20090101',
    '20120701',
    '20150701',
    '20220101']
In Julian days:
[   2441499.5,
    2441683.5,
    2442048.5,
    2442413.5,
    2442778.5,
    2443144.5,
    2443509.5,
    2443874.5,
    2444239.5,
    2444786.5,
    2445151.5,
    2445516.5,
    2446247.5,
    2447161.5,
    2447892.5,
    2448257.5,
    2448804.5,
    2449169.5,
    2449534.5,
    2450083.5,
    2450630.5,
    2451179.5,
    2453736.5,
    2454832.5,
    2456109.5,
    2457204.5,
    2457754.5,
    2459580.5]
"""
import numpy as np
import pymath.quaternion as quaternion
from pymath.common import xyz2ptr,ptr2xyz
from datetime import datetime

DEPS=np.finfo(np.double).eps
OBLIQUITY=np.deg2rad(23.439291111)
ANNOUNCED_LEAP_SECONDS_JD=np.double([
    2441499.5,
    2441683.5,
    2442048.5,
    2442413.5,
    2442778.5,
    2443144.5,
    2443509.5,
    2443874.5,
    2444239.5,
    2444786.5,
    2445151.5,
    2445516.5,
    2446247.5,
    2447161.5,
    2447892.5,
    2448257.5,
    2448804.5,
    2449169.5,
    2449534.5,
    2450083.5,
    2450630.5,
    2451179.5,
    2453736.5,
    2454832.5,
    2456109.5,
    2457204.5,
    2459580.5])

def earth_radius(lat):
    """Calculate Earth radius at given latitude.
lat - latitude, in rad

Return:
R   - radius, in km
"""
    r1 = 6378.137 # radius at equator
    r2 = 6356.752 # radius at poles
    return np.sqrt(((r1**2.*np.cos(lat))**2. + (r2**2.*np.sin(lat))**2.) / ((r1*np.cos(lat))**2. + (r2*np.sin(lat))**2.))
def deg2hms(deg):
    deg = np.mod(deg, 360.0)
    h = np.floor(deg / 15.0)
    m = np.floor((deg/15.0 - h)*60.0)
    s = ((deg/15.0 - h)*60.0 - m) * 60
    return h, m, s

def deg2dms(deg):
    deg = np.mod(deg, 360.0)
    d = np.floor(deg)
    m = np.floor((deg - d)*60.0)
    s = ((deg - d)*60.0 - m) * 60
    return d, m, s

def isoformat2datetime(isoformat_datetime_str):
    """Convert datetime string in isoformat to datetime object.

"""
    if '.' in isoformat_datetime_str:
        return datetime.strptime(isoformat_datetime_str,r'%Y-%m-%dT%H:%M:%S.%f')
    else:
        return datetime.strptime(isoformat_datetime_str,r'%Y-%m-%dT%H:%M:%S')

def isoformat2timestamp(isoformat_datetime_str):
    """Convert datetime string in isoformat to unix timestamp.

"""
    return (isoformat2datetime(isoformat_datetime_str)-\
        isoformat2datetime("1970-01-01T00:00:00.0")).total_seconds()

def convert_csys(phi,theta,from_csys=None,to_csys='ecliptic'):
    if from_csys.lower().startswith('eq'):
        if to_csys.lower().startswith('eq'):
            return phi,theta
        elif to_csys.lower().startswith('ec'):
            return equ2ec(phi,theta)
        elif to_csys.lower().startswith('ga'):
            return equ2ga(phi,theta)
    elif from_csys.lower().startswith('ec'):
        if to_csys.lower().startswith('eq'):
            return ec2equ(phi,theta)
        elif to_csys.lower().startswith('ec'):
            return phi,theta
        elif to_csys.lower().startswith('ga'):
            return equ2ga(*tuple(ec2equ(phi,theta)))
    elif from_csys.lower().startswith('ga'):
        if to_csys.lower().startswith('eq'):
            return ga2equ(phi,theta)
        elif to_csys.lower().startswith('ec'):
            return equ2ec(*tuple(ga2equ(phi,theta)))
        elif to_csys.lower().startswith('ga'):
            return phi,theta
    assert False, 'unrecognized coordinate systems.'

def equ2ec(ra,dec):
    """Convert equatorial coordinates to ecliptic coordinates.

"""
    u=quaternion.rotate(\
        quat=[np.cos(OBLIQUITY*0.5),np.sin(-0.5*OBLIQUITY),0.0,0.0],\
        vector=np.array(ptr2xyz(ra,dec,1.0)))
    l,b,rho = xyz2ptr(*tuple(u))
    if np.isscalar(ra):
        if (ra<0) & (l>np.pi):
            l = l-2.0*np.pi
    else:
        if (ra<0).any():
            idx = (l>np.pi)
            l[idx] = l[idx] - 2.0*np.pi
    return l,b

def ec2equ(l,b):
    """Convert ecliptical coordinates to equatorial coordinates.

"""
    u=quaternion.rotate(\
        quat=[np.cos(OBLIQUITY*0.5),np.sin(0.5*OBLIQUITY),0.0,0.0],\
        vector=np.array(ptr2xyz(l,b,1.0)))
    ra,dec,rho = xyz2ptr(*tuple(u))
    if np.isscalar(l):
        if (l<0) & (ra>np.pi):
            ra = ra-2.0*np.pi
    else:
        if (l<0).any():
            idx = (ra>np.pi)
            ra[idx] = ra[idx] - 2.0*np.pi
    return ra,dec

def equ2ga(ra,dec):
    """Convert equatorial coordinates to Galactic coordinates.

(J2000.0)
"""
    pole_ra = 192.859508 * np.pi / 180.0
    pole_dec = 27.128336 * np.pi / 180.0
    posangle = (122.932-90.0) * np.pi / 180.0
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    b = np.arcsin(np.cos(ra-pole_ra)*cos_dec*np.cos(pole_dec)+\
        sin_dec*np.sin(pole_dec))
    l = np.arctan2((sin_dec-np.sin(b)*np.sin(pole_dec)),\
        (cos_dec*np.sin(ra-pole_ra)*np.cos(pole_dec)))+posangle;
    l = np.mod(l,2.0*np.pi);
    if np.isscalar(ra):
        if (ra<0) & (l>np.pi):
            l = l-2.0*np.pi
    else:
        if (ra<0).any():
            idx = (l>np.pi)
            l[idx] = l[idx] - 2.0*np.pi
    return l,b

def ga2equ(l,b):
    """Convert Galactic coordinates to equatorial coordinates.

(J2000.0)
"""
    pole_ra = 192.859508 * np.pi / 180.0
    pole_dec = 27.128336 * np.pi / 180.0
    posangle = (122.932-90.0) * np.pi / 180.0
    sin_b = np.sin(b)
    cos_b = np.cos(b)
    sin_pole_dec = np.sin(pole_dec)
    cos_pole_dec = np.cos(pole_dec)
    sin_posangle = np.sin(l-posangle)
    ra = np.arctan2((cos_b*np.cos(l-posangle)),\
        (sin_b*cos_pole_dec-sin_pole_dec*cos_b*sin_posangle))+pole_ra;
    dec = np.arcsin(cos_pole_dec*cos_b*sin_posangle + sin_b*sin_pole_dec);
    ra = np.mod(ra,2.0*np.pi);
    if np.isscalar(l):
        if (l<0) & (ra>np.pi):
            ra = ra-2.0*np.pi
    else:
        if (l<0).any():
            idx = (ra>np.pi)
            ra[idx] = ra[idx] - 2.0*np.pi
    return ra,dec

def utc2tdb(UTC):
    """Convert UTC date to TDB date.

UTC is UTC date in Julian day.
"""
    TDT = utc2tdt(UTC)
    JC = (UTC - 2451545.0)/36525.0
    g = 2.0*np.pi*(357.528 + 35999.05 * JC)/360.0
    TDB = TDT + 0.001658 * np.sin(g + 0.0167*np.sin(g)) / 86400.0
    return TDB

def unix2utc(t):
    """Convert UNIX time in seconds to UTC date in Julian days.

"""
    return t/86400.0 + 2440587.5

def utc2unix(UTC):
    """Convert UTC date in Julian days to UNIX time in seconds.

"""
    return (UTC-2440587.5)*86400.0

def utc2tdt(UTC):
    """Convert UTC time in Julian days to TDT time in Julian days

"""
    return utc2tai(UTC) + 32.184/86400.0

def leap_seconds_slow(UTC):
    UTC=np.array(UTC,ndmin=1)
    dims=list(UTC.shape)
    dims.append(1)
    UTC=UTC.reshape(tuple(dims))
    ndim=UTC.ndim
    return np.sum(np.uint8(np.greater_equal(UTC,\
        np.array(ANNOUNCED_LEAP_SECONDS_JD,ndmin=ndim))),axis=ndim-1)

def leap_seconds(UTC):
    """Return total leap seconds for a given Julian day in UTC.

"""
    return np.uint8(np.greater_equal(UTC,2441499.5))+\
        np.uint8(np.greater_equal(UTC,2441683.5))+\
        np.uint8(np.greater_equal(UTC,2442048.5))+\
        np.uint8(np.greater_equal(UTC,2442413.5))+\
        np.uint8(np.greater_equal(UTC,2442778.5))+\
        np.uint8(np.greater_equal(UTC,2443144.5))+\
        np.uint8(np.greater_equal(UTC,2443509.5))+\
        np.uint8(np.greater_equal(UTC,2443874.5))+\
        np.uint8(np.greater_equal(UTC,2444239.5))+\
        np.uint8(np.greater_equal(UTC,2444786.5))+\
        np.uint8(np.greater_equal(UTC,2445151.5))+\
        np.uint8(np.greater_equal(UTC,2445516.5))+\
        np.uint8(np.greater_equal(UTC,2446247.5))+\
        np.uint8(np.greater_equal(UTC,2447161.5))+\
        np.uint8(np.greater_equal(UTC,2447892.5))+\
        np.uint8(np.greater_equal(UTC,2448257.5))+\
        np.uint8(np.greater_equal(UTC,2448804.5))+\
        np.uint8(np.greater_equal(UTC,2449169.5))+\
        np.uint8(np.greater_equal(UTC,2449534.5))+\
        np.uint8(np.greater_equal(UTC,2450083.5))+\
        np.uint8(np.greater_equal(UTC,2450630.5))+\
        np.uint8(np.greater_equal(UTC,2451179.5))+\
        np.uint8(np.greater_equal(UTC,2453736.5))+\
        np.uint8(np.greater_equal(UTC,2454832.5))+\
        np.uint8(np.greater_equal(UTC,2456109.5))+\
        np.uint8(np.greater_equal(UTC,2457204.5))+\
        np.uint8(np.greater_equal(UTC,2459580.5))

def utc2tai(UTC):
    """Convert UTC time in Julian days to TAI time in Julian days

"""
    return UTC+leap_seconds(UTC)/86400.0

def tdt2byr(TDT):
    """Convert TDT in Julian days to Besselian years.

"""
    return 1900+(TDT-2415020.31352)/365.242198781

def jd2datetime(jd):
    """Convert JD to Python datetime.
"""
    return datetime.utcfromtimestamp((jd-datetime2jd(datetime.utcfromtimestamp(0)))*86400.0)

def jd2unix(jd):
    """Convert JD to Unix timestamp.
"""
    return (jd - datetime2jd(datetime.utcfromtimestamp(0)))*86400.0

def datetime2jd(date):
    """Convert a Python datetime object to Julian days

"""
    return unix2utc((date-datetime.utcfromtimestamp(0.0)).total_seconds())

def dut1(UTC):
    """Predict DUT1=UT1-UTC, in seconds.

UTC is UTC time in Julian days.
"""
    byr=tdt2byr(utc2tdt(UTC))
    B = 2.0*np.pi*(byr-np.fix(byr))
    d = 0.022*np.sin(B)-\
        0.012*np.cos(B)-\
        0.006*np.sin(2.0*B)+\
        0.007*np.cos(2.0*B)
    return -0.5805-0.001*(UTC-2457143.5)-d+leap_seconds(UTC)-25.0

def nutation_matrix(UTC):
    """Return the nutation matrix for the given time.

Correct *only the nutation effect* between J2000 equatorial
coordinates and the instantaneous equatorial coordinates.
The returned nutation matrix should be left-multiplied to
coordinates that been corrected by precession matrix first.

Input:
UTC is UTC time in Julian days.

Reference:
1. http://www.navipedia.net/index.php/ICRF_to_CEP (The nutation matrix
   defined in this page is wrong!)
2. http://en.wikipedia.org/wiki/Barycentric_Dynamical_Time
"""
    TDB=utc2tdb(UTC)
    T=(TDB-2451545.0)/36525.0
    k1=np.double([0,0,0,0,0,1,0,0,1,0,-1,0,-1,1,0,-1,-1,1,-2,-2,0,2,2,1,0,0,\
        -1,0,0,-1,0,1,0,2,-1,1,0,0,1,0,-2,0,2,1,1,0,0,2,1,1,0,0,1,2,0,1,1,-1,\
        0,1,3,-2,1,-1,1,-1,0,-2,2,3,1,0,1,1,1,0,0,0,1,1,1,1,2,0,0,-2,2,0,0,0,\
        0,1,3,-2,-1,0,0,-1,2,2,2,2,1,-1,-1,0])
    k2=np.double([0,0,0,0,-1,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,\
        2,0,1,0,-1,0,0,0,-1,0,1,1,0,0,0,0,0,0,-1,0,-1,0,0,1,0,0,1,1,-1,-1,-1,\
        -1,0,0,0,0,0,0,-2,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,-1,0,0,1,1,\
        0,0,0,0,1,0,1,0,0,0,-1,0,-1,1])
    k3=np.double([0,2,2,0,0,0,2,2,2,2,0,2,2,0,0,2,0,2,0,2,2,2,0,2,2,2,2,0,2,0,\
        0,0,0,-2,2,2,2,2,0,2,0,0,2,0,2,0,2,2,0,0,0,0,-2,0,2,0,0,2,2,2,2,2,2,2,\
        0,2,2,0,0,0,2,2,0,2,0,0,2,-2,-2,-2,2,0,0,2,2,2,2,2,-2,4,0,2,2,2,0,-2,2,\
        4,0,0,2,-2,0,0,0,0])
    k4=np.double([0,-2,0,0,0,0,-2,0,0,-2,2,-2,0,0,2,2,0,0,2,0,2,0,0,-2,0,-2,0,\
        0,-2,2,0,-2,0,0,2,2,0,2,-2,0,2,2,-2,2,-2,-2,-2,0,0,-1,1,-2,0,-2,-2,0,\
        -1,2,2,0,0,0,0,4,0,-2,-2,0,0,0,0,1,2,2,-2,2,-2,2,2,-2,-2,-4,-4,4,-1,\
        4,2,0,0,-2,0,-2,-2,2,0,2,0,0,-2,2,-2,0,-2,1,2,1])
    k5=np.double([1,2,2,2,0,0,2,1,2,2,0,1,2,1,0,2,1,1,0,1,2,2,0,2,0,0,1,0,2,1,\
        1,1,1,0,1,2,2,1,0,2,1,1,2,0,1,1,1,1,0,0,0,0,0,1,1,0,0,2,2,2,2,2,0,2,2,\
        1,1,1,1,0,2,2,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2,1,1,2,2,2,2,2,2,1,1,2,0,0,\
        1,1,0,1,1,0])
    A0=np.double([-171996.0,-13187.0,-2274.0,2062.0,-1426.0,712.0,-517.0,\
        -386.0,-301.0,217.0,158.0,129.0,123.0,63.0,63.0,-59.0,-58.0,-51.0,\
        -48.0,46.0,-38.0,-31.0,29.0,29.0,26.0,-22.0,21.0,17.0,-16.0,16.0,\
        -15.0,-13.0,-12.0,11.0,-10.0,-8.0,-7.0,-7.0,-7.0,7.0,-6.0,-6.0,6.0,\
        6.0,6.0,-5.0,-5.0,-5.0,5.0,-4.0,-4.0,-4.0,4.0,4.0,4.0,-3.0,-3.0,\
        -3.0,-3.0,-3.0,-3.0,-3.0,3.0,-2.0,-2.0,-2.0,-2.0,-2.0,2.0,2.0,2.0,\
        2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,\
        -1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,\
        1.0,1.0,1.0,1.0,1.0])
    A1=np.double([-174.2,-1.6,-0.2,0.2,3.4,0.1,1.2,-0.4,0.0,-0.5,0.0,0.1,0.0,\
        0.1,0.0,0.0,-0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.1,0.1,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    B0=np.double([92025.0,5736.0,977.0,-895.0,54.0,-7.0,224.0,200.0,129.0,\
        -95.0,-1.0,-70.0,-53.0,-33.0,-2.0,26.0,32.0,27.0,1.0,-24.0,16.0,13.0,\
        -1.0,-12.0,-1.0,0.0,-10.0,0.0,7.0,-8.0,9.0,7.0,6.0,0.0,5.0,3.0,3.0,\
        3.0,0.0,-3.0,3.0,3.0,-3.0,0.0,-3.0,3.0,3.0,3.0,0.0,0.0,0.0,0.0,0.0,\
        -2.0,-2.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,-1.0,\
        0.0,-1.0,-1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,\
        0.0,0.0,0.0,0.0,0.0,-1.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,\
        0.0,0.0,0.0,0.0])
    B1=np.double([8.9,-3.1,-0.5,0.5,-0.1,0.0,-0.6,0.0,-0.1,0.3,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T2 = T**2.0
    T3 = T**3.0
    alpha1 =  485866.733 + (1325.0*360.0*3600.0 +  715922.633)*T + 31.310*T2 + 0.064*T3
    alpha2 = 1287099.804 + (  99.0*360.0*3600.0 + 1292581.224)*T -  0.577*T2 - 0.012*T3
    alpha3 =  335778.877 + (1342.0*360.0*3600.0 +  295263.137)*T - 13.257*T2 + 0.011*T3
    alpha4 = 1072261.307 + (1236.0*360.0*3600.0 + 1105601.328)*T -  6.891*T2 + 0.019*T3
    alpha5 =  450160.280 + (  -5.0*360.0*3600.0 +  482890.539)*T +  7.455*T2 + 0.008*T3
    alpha = np.deg2rad((alpha1*k1+alpha2*k2+alpha3*k3+alpha4*k4+alpha5*k5)/3600.0)
    epsilon = np.squeeze(np.deg2rad(23.0+26.0/60.0+21.448/3600.0 - 46.8150*T/3600.0 - 0.00059*T2/3600.0 + \
        0.001813*T3/3600.0))
    dphi = np.deg2rad(np.sum((A0+A1*T)*np.sin(alpha)*1e-4/3600.0))
    deps = np.deg2rad(np.sum((B0+B1*T)*np.cos(alpha)*1e-4/3600.0))
    N = np.array([[np.cos(dphi), -np.cos(epsilon)*np.sin(dphi), -np.sin(epsilon)*np.sin(dphi)],\
        [np.cos(epsilon+deps)*np.sin(dphi), \
        np.cos(epsilon+deps)*np.cos(epsilon)*np.cos(dphi)+np.sin(epsilon+deps)*np.sin(epsilon),\
        np.cos(epsilon+deps)*np.sin(epsilon)*np.cos(dphi)-np.sin(epsilon+deps)*np.cos(epsilon)],\
        [np.sin(epsilon+deps)*np.sin(dphi), \
        np.sin(epsilon+deps)*np.cos(epsilon)*np.cos(dphi)-np.cos(epsilon+deps)*np.sin(epsilon),\
        np.sin(epsilon+deps)*np.sin(epsilon)*np.cos(dphi)+np.cos(epsilon+deps)*np.cos(epsilon)]])
    return N

def nutation_quaternion(UTC):
    """Return the nutation quaternion for the given time.

Correct *only the nutation effect* between J2000 equatorial
coordinates and the instantaneous equatorial coordinates.
The returned nutation matrix should be left-multiplied to
coordinates that been corrected by precession matrix first.

Input:
UTC is UTC time in Julian days.

Reference:
1. http://www.navipedia.net/index.php/ICRF_to_CEP (The nutation matrix
   defined in this page is wrong!)
2. http://en.wikipedia.org/wiki/Barycentric_Dynamical_Time
"""
    if np.isscalar(UTC):
        ndim = 0
    else:
        UTC = np.reshape(UTC,(np.size(UTC),1))
        ndim = 1
    TDB=utc2tdb(UTC)
    T=(TDB-2451545.0)/36525.0
    k1=np.double([0,0,0,0,0,1,0,0,1,0,-1,0,-1,1,0,-1,-1,1,-2,-2,0,2,2,1,0,0,\
        -1,0,0,-1,0,1,0,2,-1,1,0,0,1,0,-2,0,2,1,1,0,0,2,1,1,0,0,1,2,0,1,1,-1,\
        0,1,3,-2,1,-1,1,-1,0,-2,2,3,1,0,1,1,1,0,0,0,1,1,1,1,2,0,0,-2,2,0,0,0,\
        0,1,3,-2,-1,0,0,-1,2,2,2,2,1,-1,-1,0],ndmin=ndim+1)
    k2=np.double([0,0,0,0,-1,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,\
        2,0,1,0,-1,0,0,0,-1,0,1,1,0,0,0,0,0,0,-1,0,-1,0,0,1,0,0,1,1,-1,-1,-1,\
        -1,0,0,0,0,0,0,-2,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,-1,0,0,1,1,\
        0,0,0,0,1,0,1,0,0,0,-1,0,-1,1],ndmin=ndim+1)
    k3=np.double([0,2,2,0,0,0,2,2,2,2,0,2,2,0,0,2,0,2,0,2,2,2,0,2,2,2,2,0,2,0,\
        0,0,0,-2,2,2,2,2,0,2,0,0,2,0,2,0,2,2,0,0,0,0,-2,0,2,0,0,2,2,2,2,2,2,2,\
        0,2,2,0,0,0,2,2,0,2,0,0,2,-2,-2,-2,2,0,0,2,2,2,2,2,-2,4,0,2,2,2,0,-2,2,\
        4,0,0,2,-2,0,0,0,0],ndmin=ndim+1)
    k4=np.double([0,-2,0,0,0,0,-2,0,0,-2,2,-2,0,0,2,2,0,0,2,0,2,0,0,-2,0,-2,0,\
        0,-2,2,0,-2,0,0,2,2,0,2,-2,0,2,2,-2,2,-2,-2,-2,0,0,-1,1,-2,0,-2,-2,0,\
        -1,2,2,0,0,0,0,4,0,-2,-2,0,0,0,0,1,2,2,-2,2,-2,2,2,-2,-2,-4,-4,4,-1,\
        4,2,0,0,-2,0,-2,-2,2,0,2,0,0,-2,2,-2,0,-2,1,2,1],ndmin=ndim+1)
    k5=np.double([1,2,2,2,0,0,2,1,2,2,0,1,2,1,0,2,1,1,0,1,2,2,0,2,0,0,1,0,2,1,\
        1,1,1,0,1,2,2,1,0,2,1,1,2,0,1,1,1,1,0,0,0,0,0,1,1,0,0,2,2,2,2,2,0,2,2,\
        1,1,1,1,0,2,2,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2,1,1,2,2,2,2,2,2,1,1,2,0,0,\
        1,1,0,1,1,0],ndmin=ndim+1)
    A0=np.double([-171996.0,-13187.0,-2274.0,2062.0,-1426.0,712.0,-517.0,\
        -386.0,-301.0,217.0,158.0,129.0,123.0,63.0,63.0,-59.0,-58.0,-51.0,\
        -48.0,46.0,-38.0,-31.0,29.0,29.0,26.0,-22.0,21.0,17.0,-16.0,16.0,\
        -15.0,-13.0,-12.0,11.0,-10.0,-8.0,-7.0,-7.0,-7.0,7.0,-6.0,-6.0,6.0,\
        6.0,6.0,-5.0,-5.0,-5.0,5.0,-4.0,-4.0,-4.0,4.0,4.0,4.0,-3.0,-3.0,\
        -3.0,-3.0,-3.0,-3.0,-3.0,3.0,-2.0,-2.0,-2.0,-2.0,-2.0,2.0,2.0,2.0,\
        2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,\
        -1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,\
        1.0,1.0,1.0,1.0,1.0],ndmin=ndim+1)
    A1=np.double([-174.2,-1.6,-0.2,0.2,3.4,0.1,1.2,-0.4,0.0,-0.5,0.0,0.1,0.0,\
        0.1,0.0,0.0,-0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.1,0.1,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],ndmin=ndim+1)
    B0=np.double([92025.0,5736.0,977.0,-895.0,54.0,-7.0,224.0,200.0,129.0,\
        -95.0,-1.0,-70.0,-53.0,-33.0,-2.0,26.0,32.0,27.0,1.0,-24.0,16.0,13.0,\
        -1.0,-12.0,-1.0,0.0,-10.0,0.0,7.0,-8.0,9.0,7.0,6.0,0.0,5.0,3.0,3.0,\
        3.0,0.0,-3.0,3.0,3.0,-3.0,0.0,-3.0,3.0,3.0,3.0,0.0,0.0,0.0,0.0,0.0,\
        -2.0,-2.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,-1.0,\
        0.0,-1.0,-1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,\
        0.0,0.0,0.0,0.0,0.0,-1.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,\
        0.0,0.0,0.0,0.0],ndmin=ndim+1)
    B1=np.double([8.9,-3.1,-0.5,0.5,-0.1,0.0,-0.6,0.0,-0.1,0.3,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],ndmin=ndim+1)
    T2 = T**2.0
    T3 = T**3.0
    alpha1 =  485866.733 + (1325.0*360.0*3600.0 +  715922.633)*T + 31.310*T2 + 0.064*T3
    alpha2 = 1287099.804 + (  99.0*360.0*3600.0 + 1292581.224)*T -  0.577*T2 - 0.012*T3
    alpha3 =  335778.877 + (1342.0*360.0*3600.0 +  295263.137)*T - 13.257*T2 + 0.011*T3
    alpha4 = 1072261.307 + (1236.0*360.0*3600.0 + 1105601.328)*T -  6.891*T2 + 0.019*T3
    alpha5 =  450160.280 + (  -5.0*360.0*3600.0 +  482890.539)*T +  7.455*T2 + 0.008*T3
    alpha = np.deg2rad((alpha1*k1+alpha2*k2+alpha3*k3+alpha4*k4+alpha5*k5)/3600.0)
    epsilon = np.squeeze(np.deg2rad(23.0+26.0/60.0+21.448/3600.0 - 46.8150*T/3600.0 - 0.00059*T2/3600.0 + \
        0.001813*T3/3600.0))
    dphi = np.deg2rad(np.sum((A0+A1*T)*np.sin(alpha)*1e-4/3600.0,axis=ndim))
    deps = np.deg2rad(np.sum((B0+B1*T)*np.cos(alpha)*1e-4/3600.0,axis=ndim))
    N11 = np.cos(dphi)
    N12 = -np.cos(epsilon)*np.sin(dphi)
    N13 = -np.sin(epsilon)*np.sin(dphi)
    N21 = np.cos(epsilon+deps)*np.sin(dphi)
    N22 = np.cos(epsilon+deps)*np.cos(epsilon)*np.cos(dphi)+np.sin(epsilon+deps)*np.sin(epsilon)
    N23 = np.cos(epsilon+deps)*np.sin(epsilon)*np.cos(dphi)-np.sin(epsilon+deps)*np.cos(epsilon)
    N31 = np.sin(epsilon+deps)*np.sin(dphi)
    N32 = np.sin(epsilon+deps)*np.cos(epsilon)*np.cos(dphi)-np.cos(epsilon+deps)*np.sin(epsilon)
    N33 = np.sin(epsilon+deps)*np.sin(epsilon)*np.cos(dphi)+np.cos(epsilon+deps)*np.cos(epsilon)
    w = 0.5*np.sqrt(1.0+N11+N22+N33)
    x = (N32-N23)/4.0/w
    y = (N13-N31)/4.0/w
    z = (N21-N12)/4.0/w
    return np.array([w,x,y,z])

def precession_matrix(UTC):
    """Return the precession matrix for the given time.

Correct *only the precession effect* between J2000 equatorial
coordinates and the instantaneous equatorial coordinates.
The returned precession matrix should be left-multiplied to
the J2000 equatorial coordinates directly.
(IAU 1976 Precession model)

Input:
UTC is UTC time in Julian days.

Reference:
http://www.navipedia.net/index.php/ICRF_to_CEP
http://en.wikipedia.org/wiki/Barycentric_Dynamical_Time
"""
    TDB   = utc2tdb(UTC)
    T     = (TDB-2451545.0)/36525.0
    z     = np.deg2rad((2306.2181*T+1.09468*(T**2.0)+0.018203*(T**3.0))/3600.0)
    theta = np.deg2rad((2004.3109*T-0.42665*(T**2.0)-0.041833*(T**3.0))/3600.0)
    zeta  = np.deg2rad((2306.2181*T+0.30188*(T**2.0)+0.017998*(T**3.0))/3600.0)
    P = np.matrix([\
        [np.cos(z)*np.cos(theta)*np.cos(zeta)-np.sin(z)*np.sin(zeta),\
        -np.cos(z)*np.cos(theta)*np.sin(zeta)-np.sin(z)*np.cos(zeta),\
        -np.cos(z)*np.sin(theta)],\
        [np.sin(z)*np.cos(theta)*np.cos(zeta)+np.cos(z)*np.sin(zeta),\
        -np.sin(z)*np.cos(theta)*np.sin(zeta)+np.cos(z)*np.cos(zeta),\
        -np.sin(z)*np.sin(theta)],\
        [np.sin(theta)*np.cos(zeta),-np.sin(theta)*np.sin(zeta),np.cos(theta)]])
    return P

def precession_quaternion(UTC):
    """Return the precession quaternion for the given time.

Correct *only the precession effect* between J2000 equatorial
coordinates and the instantaneous equatorial coordinates.
The returned precession matrix should be left-multiplied to
the J2000 equatorial coordinates directly.

Input:
UTC is UTC time in Julian days.

Reference:
http://www.navipedia.net/index.php/ICRF_to_CEP
http://en.wikipedia.org/wiki/Barycentric_Dynamical_Time
"""
    TDB = utc2tdb(UTC)
    T = (TDB-2451545.0)/36525.0
    T2 = T**2.0
    T3 = T**3.0
    z     = np.deg2rad((2306.2181*T+1.09468*T2+0.018203*T3)/3600.0)
    theta = np.deg2rad((2004.3109*T-0.42665*T2-0.041833*T3)/3600.0)
    zeta  = np.deg2rad((2306.2181*T+0.30188*T2+0.017998*T3)/3600.0)
    P11 =  np.cos(z)*np.cos(theta)*np.cos(zeta)-np.sin(z)*np.sin(zeta)
    P12 = -np.cos(z)*np.cos(theta)*np.sin(zeta)-np.sin(z)*np.cos(zeta)
    P13 = -np.cos(z)*np.sin(theta)
    P21 =  np.sin(z)*np.cos(theta)*np.cos(zeta)+np.cos(z)*np.sin(zeta)
    P22 = -np.sin(z)*np.cos(theta)*np.sin(zeta)+np.cos(z)*np.cos(zeta)
    P23 = -np.sin(z)*np.sin(theta)
    P31 =  np.sin(theta)*np.cos(zeta)
    P32 = -np.sin(theta)*np.sin(zeta)
    P33 =  np.cos(theta)
    w = 0.5*np.sqrt(1.0+P11+P22+P33)
    x = (P32-P23)/4.0/w
    y = (P13-P31)/4.0/w
    z = (P21-P12)/4.0/w
    return np.array([w,x,y,z])

def gmst(UTC):
    """Greenwich mean sidereal time (in degrees) at given time.

Input:
UTC - UTC time in Julian days.
"""
    UTC   = np.array(UTC)
    # JD    = UTC+dut1(UTC)/86400.0
    JD    = UTC
    JDmin = np.floor(JD)-0.5
    JDmax = np.floor(JD)+0.5
    try:
        JD0 = np.empty(UTC.shape)
        JD0[JD>=JDmin] = JDmin[JD>=JDmin]
        JD0[JD>=JDmax] = JDmax[JD>=JDmax]
    except:
        if JD>=JDmin:
            JD0=JDmin
        if JD>=JDmax:
            JD0=JDmax
    H = (JD-JD0)*360.0
    T = (JD0-2451545.0)/36525.0
    return np.mod(1.002737909350795*H+\
        (6.0+41.0/60.0+50.54841/3600.0)*15.0+\
        (8640184.812866*T+0.093104*(T**2.0)-6.21*(T**3.0))/3600.0*15.0,360.0)

def sidereal_matrix(UTC):
    """Return the sidereal matrix for given time.

The sidereal matrix defines a rotation around the CEP pole of angle ThetaG.
The angle ThetaG is the Greenwich Apparent sidereal time of the given time.

Input:
UTC is UTC time in Julian days.

Reference:
http://www.navipedia.net
"""
    N=nutation_matrix(UTC)
    alphaE=np.arctan2(N[0,1],N[0,0])
    ThetaG=alphaE + np.deg2rad(gmst(UTC))
    return np.matrix([[np.cos(ThetaG),np.sin(ThetaG),0.0],\
        [-np.sin(ThetaG),np.cos(ThetaG),0.0],\
        [0.0,0.0,1.0]])

def sidereal_quaternion(UTC):
    """Return the sidereal quaternion for given time.

The sidereal matrix defines a rotation around the CEP pole of angle ThetaG.
The angle ThetaG is the Greenwich Apparent sidereal time of the given time.

Input:
UTC is UTC time in Julian days.

Reference:
http://www.navipedia.net
"""
    qN=nutation_quaternion(UTC)
    alphaE=np.arctan2(2.0*(qN[1]*qN[2]-qN[0]*qN[3]),qN[0]**2.0+qN[1]**2.0-qN[2]**2.0-qN[3]**2.0)
    ThetaG=alphaE + np.deg2rad(gmst(UTC))
    return np.array([np.cos(ThetaG*0.5),0.0,0.0,-np.sin(ThetaG*0.5)])

def horizontal_quaternion(lon,lat):
    """Convert geodetic coordinate vector to local horizontal coordinate vector.

Equivalently rotate horizontal coordinate frame to geodetic coordinate frame.
"""
    return 0.5*np.sqrt(2.0)*np.double([\
         np.cos(0.5*lon)*(np.cos(0.5*lat)+np.sin(0.5*lat)),\
        -np.sin(0.5*lon)*(np.sin(0.5*lat)-np.cos(0.5*lat)),\
        -np.cos(0.5*lon)*(np.cos(0.5*lat)-np.sin(0.5*lat)),\
        -np.sin(0.5*lon)*(np.sin(0.5*lat)+np.cos(0.5*lat))])

def geo2hor(lon0,lat0,lon,lat):
    """Convert geographic coordinates to horizontal coordinates.

lon0 and lat0 are geographic coordinates of observer's location.
lon and lat are geographic coordinates to convert.

Return:
alt and az are altitude and azimuth in horizontal coordinates.
"""
    rgeo=np.array(ptr2xyz(lon,lat))
    lon0=0.5*lon0
    lat0=0.5*lat0
    qg2h=0.5*np.sqrt(2.0)*np.double([\
        np.cos(lon0)*(np.cos(lat0)+np.sin(lat0)),\
        -np.sin(lon0)*(np.sin(lat0)-np.cos(lat0)),\
        -np.cos(lon0)*(np.cos(lat0)-np.sin(lat0)),\
        -np.sin(lon0)*(np.sin(lat0)+np.cos(lat0))])
    rhor=quaternion.rotate(quat=qg2h,vector=rgeo)
    az, alt, _=xyz2ptr(*tuple(rhor))
    return az, alt

def hor2geo(lon0, lat0, az, alt):
    """Convert horizontal coordinates to geographic coordinates.

lon0 and lat0 are geographic coordinates of observer's location.
alt and az are altitude and azimuth in horizontal coordinates.

Return:
lon and lat are geographic coordinates.
"""
    rhor=np.array(ptr2xyz(az, alt))
    lon0=0.5*lon0
    lat0=0.5*lat0
    qg2h=0.5*np.sqrt(2.0)*np.double([\
        np.cos(lon0)*(np.cos(lat0)+np.sin(lat0)),\
        -np.sin(lon0)*(np.sin(lat0)-np.cos(lat0)),\
        -np.cos(lon0)*(np.cos(lat0)-np.sin(lat0)),\
        -np.sin(lon0)*(np.sin(lat0)+np.cos(lat0))])
    rgeo=quaternion.rotate(quaternion.conjugate(qg2h), rhor)
    lon, lat, _=xyz2ptr(*tuple(rgeo))
    return lon, lat

def equ2geo(UTC,ra,dec):
    """Convert equatorial coordinates to geodetic coordinates at given time.

UTC - UTC time in JD
ra  - R.A., in rad
dec - Dec., in rad

Returns:
lon - geodetic longitude, in rad
lat - geodetic latitude, in rad
"""
    P         = precession_matrix(UTC)
    N         = nutation_matrix(UTC)
    S         = sidereal_matrix(UTC)
    requ      = ptr2xyz(ra,dec)
    qgeo      = quaternion.from_matrix(S*N*P)
    rgeo      = quaternion.rotate(qgeo, requ)
    lon,lat,_ = xyz2ptr(*tuple(rgeo))
    return lon,lat

def geo2equ(UTC,lon,lat):
    """Convert geodetic coordinates to equatorial coordinates at given time.
UTC - UTC time in JD
lon - longitude, in rad
lat - latitude, in rad

Returns:
ra  - right ascension, in rad
dec - declination, in rad
"""
    P          = precession_matrix(UTC)
    N          = nutation_matrix(UTC)
    S          = sidereal_matrix(UTC)
    rgeo       = ptr2xyz(lon, lat)
    qequ       = quaternion.from_matrix(np.linalg.inv(S*N*P))
    requ       = quaternion.rotate(qequ, rgeo)
    ra, dec, _ = xyz2ptr(*tuple(requ))
    return ra, dec

def equ2geo_fast(UTC,ra,dec):
    """Convert equatorial coordinate to geodetic coordinate.

Takes precession into account.
"""
    requ = ptr2xyz(ra,dec)
    qP = precession_quaternion(UTC)
    ThetaG = np.deg2rad(gmst(UTC))
    qS = np.array([np.cos(ThetaG*0.5),0.0,0.0,-np.sin(ThetaG*0.5)])
    rgeo = quaternion.rotate(qS,quaternion.rotate(qP,requ))
    lon,lat,_=xyz2ptr(*tuple(rgeo))
    return lon,lat

def equ2geo_veryfast(UTC,ra,dec):
    """Convert equatorial coordinate to geodetic coordinate.

Disregards both nutation and precession.
"""
    requ = ptr2xyz(ra,dec)
    ThetaG = np.deg2rad(gmst(UTC))
    qS = np.array([np.cos(ThetaG*0.5),0.0,0.0,-np.sin(ThetaG*0.5)])
    rgeo = quaternion.rotate(qS,requ)
    lon,lat,_=xyz2ptr(*tuple(rgeo))
    return lon,lat

def equ2hor(UTC,lon0,lat0,ra,dec):
    lon,lat=equ2geo(UTC,ra,dec)
    az,alt=geo2hor(lon0,lat0,lon,lat)
    return az,alt

def polar_motion_matrix():
    pass

def _mollweide_aux(theta):
    theta = np.double(theta)
    # if theta is a scalar:
    if np.isscalar(theta):
        if np.abs(np.abs(theta)-np.pi*0.5)<=DEPS:
            return np.sign(theta)*np.pi*0.5
        else:
            aux = theta
            delta = np.pi
            while np.abs(delta) > 1e-8:
                aux2 = 2.0*aux
                delta = (aux2 + np.sin(aux2) - np.pi*np.sin(theta)) \
                    / (2.0+2.0*np.cos(aux2))
                aux = aux - delta
            return aux
    # if theta is an array
    else:
        idx = (np.abs(np.abs(theta)-np.pi*0.5) > DEPS)
        # theta[~idx] = np.sign(theta[~idx])*np.pi*0.5
        aux = np.array(theta)
        delta = np.pi * np.ones(theta.shape)
        while (np.abs(delta[idx]) > 1e-8).any():
            aux2 = 2.0*aux[idx]
            delta[idx] = (aux2 + np.sin(aux2) - np.pi*np.sin(theta[idx])) \
                / (2.0+2.0*np.cos(aux2))
            aux[idx] = aux[idx] - delta[idx]
        return aux

def mollweide(phi, theta):
    """Mollweide Projection

Inputs:
phi   in [-pi  ,   pi]
theta in [-pi/2, pi/2]

Returns:
x in [-pi      ,  pi]
y in [-pi/2    ,pi/2]
"""
    # normalize phi and theta
##    phi   = np.mod(np.double(phi),   2.0*np.pi)
##    theta = np.mod(np.double(theta), 2.0*np.pi)
##    if np.isscalar(phi):
##        if (theta>np.pi*0.5) & (theta<np.pi*1.5):
##            phi = np.mod(phi+np.pi, 2.0*np.pi)
##            theta = np.pi - theta
##        if (theta>=np.pi*1.5):
##            theta = theta - 2.0*np.pi
##        if (phi>np.pi):
##            phi = phi-2.0*np.pi
##    else:
##        idx = np.logical_and((theta>np.pi*0.5), (theta<np.pi*1.5))
##        phi[idx] = np.mod(phi[idx]+np.pi, 2.0*np.pi)
##        theta[idx] = np.pi - theta[idx]
##        idx = theta>=np.pi*1.5
##        theta[idx] = theta[idx] - 2.0*np.pi
##        idx = phi>np.pi
##        phi[idx] = phi[idx]-2.0*np.pi
    # mollweide projection
    aux = _mollweide_aux(theta)
    x = phi * np.cos(aux)
    y = np.pi * 0.5 * np.sin(aux)
    return x, y

def imollweide(x,y):
    """Inverse Mollweide Projection
"""
    aux = np.arcsin(np.double(y)*2.0/np.pi)
    aux2 = aux*2.0
    theta = np.arcsin((aux2 + np.sin(aux2)) / np.pi)
    if np.isscalar(theta):
        if (np.abs(np.abs(aux) - np.pi*0.5) > DEPS):
            phi = np.double(x)/np.cos(aux)
        else:
            phi = 0.0
    else:
        phi = np.zeros(theta.shape)
        idx = np.abs(np.abs(aux) - np.pi*0.5) > DEPS
        phi[idx] = np.double(x[idx])/np.cos(aux[idx])
    return phi, theta
