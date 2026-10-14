__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Fresnel amplitudes of PreRefl.reflectivity_amplitudes_fresnel_external:
# the grazing-incidence (Parratt) approximation (method=1) must agree with the full Fresnel equations
# (method=0, Born & Wolf) at small grazing angles, including the phase. With n = 1 - delta + i beta the
# Parratt term is f = sqrt(theta^2 - 2 delta + 2 i beta); the wrong sign (- 2 i beta) gave the complex
# conjugate amplitude (same reflectivity, opposite phase).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from shadow4.physical_models.prerefl.prerefl import PreRefl


def amplitudes(n2, grazing_mrad, method, n1=1.0 + 0j, energy=10000.0):
    return PreRefl.reflectivity_amplitudes_fresnel_external(photon_energy_ev=energy,
                                                           refraction_index_1=numpy.array([n1]),
                                                           refraction_index_2=numpy.array([n2]),
                                                           grazing_angle_mrad=numpy.array([grazing_mrad]),
                                                           roughness_rms_A=0.0,
                                                           method=method)


class TestPreReflFresnel(unittest.TestCase):

    def test_parratt_vs_born_wolf(self):
        n2 = 1 - 7.6e-6 + 1.7e-7j      # X-ray mirror coating, n = 1 - delta + i beta
        for grazing_mrad in [0.5, 1.0, 2.0, 3.0, 3.8, 4.0, 5.0, 8.0]:
            rs_bw, rp_bw = amplitudes(n2, grazing_mrad, 0)
            rs_pa, rp_pa = amplitudes(n2, grazing_mrad, 1)
            assert_allclose(rs_pa, rs_bw, rtol=1e-3, atol=1e-6, err_msg="theta_g=%g mrad" % grazing_mrad)
            assert_allclose(rp_pa, rp_bw, rtol=1e-3, atol=1e-6, err_msg="theta_g=%g mrad" % grazing_mrad)

    def test_total_reflection_phases(self):
        # glass to air (n1=1.5), theta1=60 deg from the normal: total internal reflection, |r|=1,
        # tan(phi_s/2) = -sqrt(sin^2 - n^2)/cos, tan(phi_p/2) = -(n1/n2)^2 sqrt(sin^2 - n^2)/cos (exp(-i w t))
        n1, theta1 = 1.5, numpy.radians(60.0)
        rs, rp = amplitudes(1.0 + 0j, 1e3 * (numpy.pi / 2 - theta1), 0, n1=n1 + 0j, energy=2.0)
        kappa = numpy.sqrt(numpy.sin(theta1) ** 2 - (1 / n1) ** 2) / numpy.cos(theta1)
        assert_allclose(numpy.abs([rs[0], rp[0]]), [1.0, 1.0], atol=1e-12)
        assert_allclose(numpy.angle(rs[0]), -2 * numpy.arctan(kappa), atol=1e-12)
        assert_allclose(numpy.angle(rp[0]), -2 * numpy.arctan(n1 ** 2 * kappa), atol=1e-12)

    def test_normal_and_grazing_limits(self):
        # SHADOW pi convention: r_p = -r_s at normal incidence, r_p -> r_s -> -1 at grazing incidence
        n2 = 1.5 + 0j
        rs, rp = amplitudes(n2, 1e3 * numpy.pi / 2, 0, energy=2.0)
        assert_allclose(rs[0], (1 - 1.5) / (1 + 1.5), atol=1e-12)
        assert_allclose(rp[0], -rs[0], atol=1e-12)
        rs, rp = amplitudes(n2, 1e-3, 0, energy=2.0)
        assert_allclose([rs[0], rp[0]], [-1.0, -1.0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
