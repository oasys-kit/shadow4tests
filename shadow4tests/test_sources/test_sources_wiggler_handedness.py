__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Handedness of the wiggler radiation out of the orbit plane. At each pole, the radiation is like that of a bending
# magnet: if the electron turns toward +x (positive curvature d2x/dy2, clockwise seen from above) the radiation emitted
# above the orbit plane (Z' > 0) is right-handed (S3 < 0, for an observer looking toward the source), and below it is
# left-handed; the handedness is reversed at the poles where the electron turns toward -x. See
# test_sources_bm_handedness.py, F. Schafers, in Modern Developments in X-Ray and Neutron Optics (Springer, 2008),
# p. 30, and the Lienard-Wiechert field of an electron on a circular orbit.
#

import unittest
import numpy

from shadow4.sources.s4_electron_beam import S4ElectronBeam
from shadow4.sources.wiggler.s4_wiggler import S4Wiggler
from shadow4.sources.wiggler.s4_wiggler_light_source import S4WigglerLightSource


def wiggler_light_source(nrays=5000):
    electron_beam = S4ElectronBeam(energy_in_GeV=6.084, energy_spread=0.001, current=0.1)
    electron_beam.set_sigmas_all(sigma_x=0, sigma_y=0, sigma_xp=0, sigma_yp=0)
    magnetic_structure = S4Wiggler(magnetic_field_periodic=1, K_vertical=20.4911, period_length=0.15,
                                   number_of_periods=6, emin=50000.0, emax=50000.0, ng_e=1, ng_j=501,
                                   psi_interval_number_of_points=101, flag_interpolation=1, flag_emittance=0,
                                   shift_x_flag=4, shift_betax_flag=4)
    return S4WigglerLightSource(electron_beam=electron_beam, magnetic_structure=magnetic_structure,
                                nrays=nrays, seed=5676561)


class TestWigglerHandedness(unittest.TestCase):

    def test_handedness(self):
        light_source = wiggler_light_source()
        beam = light_source.get_beam()

        # curvature of the trajectory at the x position of the rays (no emittance: the rays start on the trajectory)
        traj = light_source._S4WigglerLightSource__result_trajectory
        x_traj, y_traj, curv = traj[0], traj[1], traj[6]
        d2x = numpy.gradient(numpy.gradient(x_traj, y_traj), y_traj)
        poles = numpy.abs(curv) > 0.5 * numpy.abs(curv).max()
        self.assertTrue(numpy.all(numpy.sign(curv[poles]) == numpy.sign(d2x[poles])))  # curv = d2x/dy2

        y, zp = beam.get_column(2, nolost=1), beam.get_column(6, nolost=1)
        P3 = beam.get_column(44, nolost=1)
        curv_rays = numpy.interp(y, y_traj, curv)
        good = (numpy.abs(curv_rays) > 0.5 * numpy.abs(curv).max()) & (numpy.abs(P3) > 0.05)
        for sign_curv in [1, -1]:
            for side, z in [("above", zp > 0), ("below", zp < 0)]:
                mask = good & z & (numpy.sign(curv_rays) == sign_curv)
                self.assertGreater(mask.sum(), 100)
                expected = -sign_curv if side == "above" else sign_curv
                self.assertTrue(numpy.all(numpy.sign(P3[mask]) == expected),
                                msg="curvature %+d, %s the orbit plane" % (sign_curv, side))


if __name__ == "__main__":
    unittest.main()
