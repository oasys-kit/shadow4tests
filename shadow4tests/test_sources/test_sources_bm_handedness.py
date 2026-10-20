__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Handedness of the bending magnet radiation out of the orbit plane. For B < 0 the electrons are deflected toward +x
# (clockwise seen from above) and the radiation emitted above the orbit plane (Z' > 0) is right-handed (S3 < 0, for
# an observer looking toward the source); below the plane it is left-handed. For B > 0 the handedness is reversed.
# The result must not depend on the sign of the radius. See F. Schafers, in Modern Developments in X-Ray and Neutron
# Optics (Springer, 2008), p. 30, and the Lienard-Wiechert field of an electron on a circular orbit.
#

import unittest
import numpy

from shadow4.sources.s4_electron_beam import S4ElectronBeam
from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import S4BendingMagnetLightSource


def bm_beam(magnetic_field, radius_sign=1, nrays=2000):
    electron_beam = S4ElectronBeam(energy_in_GeV=6.04, energy_spread=0.001, current=0.2)
    electron_beam.set_sigmas_all(sigma_x=7.8e-05, sigma_y=4.87179e-05, sigma_xp=3.6e-05, sigma_yp=1.05556e-06)
    radius = radius_sign * numpy.abs(S4BendingMagnet.calculate_magnetic_radius(magnetic_field, electron_beam.energy()))
    magnetic_structure = S4BendingMagnet(radius=radius, magnetic_field=magnetic_field, length=1e-3 * abs(radius),
                                         emin=8000.0, emax=8000.0, ng_e=100, flag_emittance=0)
    light_source = S4BendingMagnetLightSource(electron_beam=electron_beam, magnetic_structure=magnetic_structure,
                                              nrays=nrays, seed=5676561)
    return light_source.get_beam()


class TestBendingMagnetHandedness(unittest.TestCase):

    def test_handedness(self):
        for magnetic_field in [-0.8, 0.8]:
            for radius_sign in [1, -1]:
                beam = bm_beam(magnetic_field, radius_sign=radius_sign)
                x, zp = beam.get_column(1), beam.get_column(6)
                P3 = beam.get_column(44)
                msg = "B=%g, radius sign %d" % (magnetic_field, radius_sign)
                # deflection: toward +x for B < 0
                self.assertEqual(numpy.sign(x.mean()), -numpy.sign(magnetic_field), msg=msg)
                # handedness: S3 < 0 above the orbit plane for B < 0
                self.assertEqual(numpy.sign(P3[zp > 0].mean()), numpy.sign(magnetic_field), msg=msg)
                self.assertEqual(numpy.sign(P3[zp < 0].mean()), -numpy.sign(magnetic_field), msg=msg)
                self.assertTrue(numpy.all(numpy.sign(P3[zp > 0]) == numpy.sign(magnetic_field)), msg=msg)


if __name__ == "__main__":
    unittest.main()
