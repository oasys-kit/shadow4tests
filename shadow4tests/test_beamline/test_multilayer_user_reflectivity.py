__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Multilayer with reflectivities from user files (f_refl=1: R vs angle, 2: R vs energy, 3: R vs energy and angle,
# with one column R or two columns Rs, Rp): the intensities of the sigma and pi components are multiplied once by
# Rs and Rp.
#
# Previously, for f_refl=3 with one reflectivity column, the reflectivity was applied three times (intensity R^3).
#

import os
import tempfile
import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.multilayers.s4_plane_multilayer import S4PlaneMultilayer, S4PlaneMultilayerElement

RS = 0.25
RP = 0.64
GRAZING = 0.02 # rad


def get_beam():
    s = SourceGeometrical(nrays=1000, seed=5676561)
    s.set_spatial_type_point()
    s.set_angular_distribution_gaussian(sigdix=1e-5, sigdiz=1e-5)
    s.set_energy_distribution_singleline(10000.0)
    s.set_polarization(polarization_degree=0.5, phase_diff=0, coherent_beam=0) # equal sigma and pi intensities
    return s.get_beam()


def trace(beam, f_refl, file_refl):
    oe = S4PlaneMultilayer(f_refl=f_refl, file_refl=file_refl)
    element = S4PlaneMultilayerElement(optical_element=oe, input_beam=beam,
                                       coordinates=ElementCoordinates(p=1.0, q=1.0,
                                                                      angle_radial=numpy.pi / 2 - GRAZING))
    beam_out, _ = element.trace_beam()
    return beam_out


class TestMultilayerUserReflectivity(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.beam = get_beam()

    def tearDown(self):
        self.tmpdir.cleanup()

    def write_file(self, name, rows):
        filename = os.path.join(self.tmpdir.name, name)
        numpy.savetxt(filename, numpy.array(rows))
        return filename

    def check(self, beam_out, rs, rp):
        Is_in, Ip_in = [self.beam.get_column(c, nolost=1).sum() for c in (24, 25)]
        Is_out, Ip_out = [beam_out.get_column(c, nolost=1).sum() for c in (24, 25)]
        assert_allclose(Is_out / Is_in, rs, rtol=1e-6)
        assert_allclose(Ip_out / Ip_in, rp, rtol=1e-6)

    def test_f_refl_1_angle(self):
        angles_mrad = numpy.linspace(1, 40, 40)
        f = self.write_file("refl_angle.dat", [[a, RS] for a in angles_mrad])
        self.check(trace(self.beam, 1, f), RS, RS)

    def test_f_refl_2_energy(self):
        energies = numpy.linspace(5000, 15000, 11)
        f = self.write_file("refl_energy.dat", [[e, RS] for e in energies])
        self.check(trace(self.beam, 2, f), RS, RS)

    def test_f_refl_3_energy_angle(self):
        energies = numpy.linspace(5000, 15000, 11)
        angles_mrad = numpy.linspace(1, 40, 40)
        f = self.write_file("refl_2d.dat", [[e, a, RS] for e in energies for a in angles_mrad])
        self.check(trace(self.beam, 3, f), RS, RS)

    def test_f_refl_3_energy_angle_s_p(self):
        energies = numpy.linspace(5000, 15000, 11)
        angles_mrad = numpy.linspace(1, 40, 40)
        f = self.write_file("refl_2d_sp.dat", [[e, a, RS, RP] for e in energies for a in angles_mrad])
        self.check(trace(self.beam, 3, f), RS, RP)


if __name__ == "__main__":
    unittest.main()
