import numpy
from srxraylib.plot.gol import plot, set_qt

set_qt()

filename = "matsuyama_tolerances_akb.dat"
a = numpy.loadtxt(filename)
filename = "matsuyama_tolerances_akb2.dat"
a2 = numpy.loadtxt(filename)
filename = "matsuyama_tolerances_akb3.dat"
a3 = numpy.loadtxt(filename)
filename = "matsuyama_tolerances_kb.dat"
b = numpy.loadtxt(filename)
# plot(a[:, 0], a[:, 1],
#      a2[:, 0], a2[:, 1],
#      b[:, 0], b[:, 1], legend=["AKB M2","AKB M4","KB M2"],
#      xtitle="Error in roll [um]", ytitle="Focus H size FWHM [nm]")
plot(
     a2[:, 0], a2[:, 1],
     a3[:, 0], a3[:, 1],
     b[:, 0], b[:, 1],
     legend=["AKB M4 (shadow method)","AKB M4 (coefficient rotation)","KB M2"],
     xtitle="Error in roll [urad]", ytitle="Focus H size FWHM [nm]")