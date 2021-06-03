#!/usr/bin/env python
#coding=utf-8
"""Observer related helpers.

Syntax:
  humanvision.py [action] [options]

Actions:
  help        print this help message.
  synthesize  synthesize NumPy cdata into graphic files such as TIFF, PNG, ...

Options:
  -g, --gamma       `gamma` for normalizing luminance component.
  -c, --csys         coordinate system(s).
  -p, --projection   projection (mollweide, cartesian).
  -i, --input        input file.
  -o, --output       output file.

Copyright: pigsboss@github
"""
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import healpy as hp
import sys
from getopt import gnu_getopt
from pimms.base import rgb_to_hsl, hsl_to_rgb
from os import path
from astrotoys.observer.init_bbr_color import bbr_color_dtype
from scipy.interpolate import interp1d
from imageio import imwrite

pixfmt = np.dtype([
    ('r', 'f8'),
    ('g', 'f8'),
    ('b', 'f8')
])

class CIEObserver:
    """The International Commision on Illumunation (Commission internationale de l'éclairage, CIE)
defines the quantitative links between distributions of wavelengths (e.g., spectral flux), and
physiologically perceived colors in human color vision.
This object serves as a CIE standard observer, that:
  (1). maps black body radiation spectrum to RGB color space,
  (2). converts color coordinates between RGB, HSL, HSV and other color spaces.

RGB color space, or standard RGB (sRGB) color space.
Color coordinates are r(ed), g(reen) and b(lue), physically speaking, which represent integral
flux in red, green and blue bands individually weighted by corresponding photopic luminosity
functions (like bandpass filters and `human visual` sensors instead of telescope filters and CCD
that convert incoming radiance into digitized electron countings).

HSL color space is cylindrical space, where hue is the argument angle, saturation is the radius
and luminance is the height.
Hue is color. In astrophysics we often define a color function of a specific spectral flux by
(a-b)/(b-c), where a, b and c are integral flux over three bands. Here the hue coordinate in
HSL color space is also related to (r-g)/(b-r).
In HSL color space hue is the argument angle, where primary colors red, green and blue are
0, 120 and 240 degrees. Secondary colors yellow, cyan and magenta are 1:1 linear mixtures between
adjacent primary colors so their argument angles are 60, 180 and 300 degrees.
Saturation is colorfulness of a stimulus relative to its own brightness.
Grayscale images are not colorful at all so their saturations are 0.
Primary colors, secondary colors and their linear mixtures between adjacent pairs of them
are called pure colors because their radial coordinate (saturation) is 1.
Luminance is analogous to radiance intensity. It is weighted sum of intensity of all color bands.

Back to the astronomy observation context.
Typical sources such as main sequence stars are characterised by two parameters, effective
temperature Teff and magnitude m, in photometrics. To render such sources for human vision,
we can take the following steps:
  (1). map black body temperature to RGB color space with CIE standard observer,
  (2). calculate hue and saturation from r, g and b coordinates,
  (3). normalize and map the apparent magnitude of the source to [0,1] range and yield the
       relative luminance.
"""
    def __init__(self, method='cubic', CMF='10deg'):
        modpath = path.split(path.normpath(path.abspath(path.realpath(__file__))))[0]
        t = np.copy(np.memmap(
            path.join(modpath, 'bbr_color_{}.npy'.format(CMF)),
            dtype=bbr_color_dtype,
            mode='r'))
        self.bbr_r = interp1d(t['K'], t['R'], kind=method, fill_value='extrapolate')
        self.bbr_g = interp1d(t['K'], t['G'], kind=method, fill_value='extrapolate')
        self.bbr_b = interp1d(t['K'], t['B'], kind=method, fill_value='extrapolate')
    def K_to_rgb(self, K):
        return np.clip(self.bbr_r(K),0,1), np.clip(self.bbr_g(K),0,1), np.clip(self.bbr_b(K),0,1)

def synthesize(input_file, gamma, output_file, projection, csys):
    pixelization, response, resolstr = path.splitext(path.splitext(path.split(input_file)[1])[0])[0].split('_') # naming pattern: pixelization_response_resolstr.cdata.npy
    if pixelization.lower() in ['healpix', 'hpx']:
        print('Pixelization scheme: HEALPix')
        nside = int(resolstr[1:])
        hpxdata = np.memmap(input_file, mode='r', dtype=pixfmt)
        if projection.lower().startswith('c'):
            r = hp.cartview(hpxdata['r'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
            g = hp.cartview(hpxdata['g'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
            b = hp.cartview(hpxdata['b'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
        elif projection.lower().startswith('m'):
            r = hp.mollview(hpxdata['r'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
            g = hp.mollview(hpxdata['g'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
            b = hp.mollview(hpxdata['b'], nest=True, xsize=6*nside, coord=csys, return_projected_map=True)
        plt.close('all')
        r = np.clip(r.data, 0., None) ## healpy.cartview and healpy.mollview return masked array.
        g = np.clip(g.data, 0., None) ## healpy.cartview and healpy.mollview return masked array.
        b = np.clip(b.data, 0., None) ## healpy.cartview and healpy.mollview return masked array.
        shape = r.shape
    elif pixelization.lower() == 'p3g':
        print('Pixelization scheme: P3G')
        shape = tuple(map(int, resolstr.split('x')))
        rgb = np.memmap(input_file, mode='r', shape=shape, dtype=pixfmt)
        r = rgb['r']
        g = rgb['g']
        b = rgb['b']
    rgb_max = np.max([r,g,b])
    RGB = np.zeros(shape+(3,))
    h, s, l = rgb_to_hsl(r[:]/rgb_max, g[:]/rgb_max, b[:]/rgb_max)
    RGB[:,:,0], RGB[:,:,1], RGB[:,:,2] = hsl_to_rgb(h, s, l**gamma)
    if path.splitext(output_file)[1].lower() in ['.png', '.jpg', '.gif', '.bmp']:
        imwrite(output_file, np.uint8(np.clip(RGB[::-1,:,:], 0., 1.)*255))
    elif path.splitext(output_file)[1].lower() in ['.tif', '.tiff']:
        imwrite(output_file, np.uint16(np.clip(RGB[::-1,:,:], 0., 1.)*65535))

def test():
    obs = CIEObserver()
    K,_= np.meshgrid(np.arange(1000,40000,20), np.ones(512))
    im = np.empty(K.shape+(3,))
    im[:,:,0], im[:,:,1], im[:,:,2] = obs.K_to_rgb(K)
    plt.imshow(im,origin='lower',extent=(1000,40000,1,20000))
    plt.xlabel('BB Temperature, in Kelvin')
    plt.yticks([])
    plt.show()
    return

if __name__ == '__main__':
    opts, args = gnu_getopt(sys.argv[1:], 'g:c:p:i:o:', [
        'gamma=',
        'csys=',
        'projection=',
        'input=',
        'output='
    ])
    action = args[0]
    if action.lower() in ['help']:
        print(__doc__)
        sys.exit()
    elif action.lower() in ['synthesize']:
        projection = 'cart'
        csys = None
        gamma = 0.5
        for opt, val in opts:
            if opt.lower() in ['-i', '--input']:
                input_file = path.normpath(path.abspath(path.realpath(val)))
                assert path.isfile(input_file), 'input file does not exists.'
            elif opt.lower() in ['-o', '--output']:
                output_file = path.normpath(path.abspath(path.realpath(val)))
            elif opt.lower() in ['-g', '--gamma']:
                gamma = float(val)
            elif opt.lower() in ['-p', '--projection']:
                projection = val
            elif opt.lower() in ['-c', '--csys']:
                csys = val
        synthesize(input_file, gamma, output_file, projection, csys)
