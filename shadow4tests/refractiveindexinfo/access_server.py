
# see https://refractiveindex.info and  https://github.com/toftul/refractiveindex/

import numpy

from refractiveindex import RefractiveIndexMaterial

mat = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Franta')


WAVELENGTH_MICRONS = numpy.linspace(0.025, 125, 100)
WAVELENGTH_NM = WAVELENGTH_MICRONS * 1000

N = []
EPSILON = []
K = []
for wavelength_nm in WAVELENGTH_NM:
    EPSILON.append(mat.get_epsilon(wavelength_nm))
    N.append(mat.get_refractive_index(wavelength_nm))
    K.append(mat.get_extinction_coefficient(wavelength_nm))


from srxraylib.plot.gol import plot
plot(WAVELENGTH_MICRONS, N,
     WAVELENGTH_MICRONS, K, legend=["n","k"], xtitle="Wavelength [um]")