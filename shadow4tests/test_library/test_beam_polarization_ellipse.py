__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Polarization ellipse (S4Beam.get_polarization_ellipse and S4Beam.get_polarization_ellipse_curve), computed from the
# Stokes columns 30-33, checked against the standard Jones vectors (Ex, Ey) = (Es, -Ep) of the rays: the real field
# over one period, E_j(tau) = Re[E_j exp(-i tau)], in the right-handed frame (u_s, -u_p, v), for an observer looking
# toward the source (C. Detlefs et al., Eur. Phys. J. Special Topics 208, 359 (2012)).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from shadow4.beam.s4_beam import S4Beam
from shadow4.tools.arrayofvectors import vector_cross


def make_beam(a_s, a_p, phs, php):
    b = S4Beam.initialize_as_pencil(N=a_s.size)
    v = b.get_columns([4, 5, 6]).T
    e_S = numpy.tile([1.0, 0, 0], (a_s.size, 1))
    b.set_jones_components(a_s * numpy.exp(1j * phs), a_p * numpy.exp(1j * php), e_S=e_S, e_P=vector_cross(e_S, v))
    return b


def random_beam(n=200, seed=11):
    rng = numpy.random.default_rng(seed)
    a_s, a_p = rng.uniform(0.0, 1, n), rng.uniform(0.0, 1, n)
    phs, php = rng.uniform(0, 2 * numpy.pi, n), rng.uniform(0, 2 * numpy.pi, n)
    return make_beam(a_s, a_p, phs, php)


class TestBeamPolarizationEllipse(unittest.TestCase):

    def test_against_jones(self):
        # ellipses of all the rays (flag_return=1) against the real field from the Jones vectors
        b = random_beam()
        ell = b.get_polarization_ellipse(flag_return=1, nolost=0)
        Ex, Ey = b.get_jones_components(nolost=0, use_standard=1)

        S = [abs(Ex) ** 2 + abs(Ey) ** 2, abs(Ex) ** 2 - abs(Ey) ** 2,
             2 * numpy.real(Ex * numpy.conj(Ey)), 2 * numpy.imag(numpy.conj(Ex) * Ey)]
        assert_allclose(ell["S"], S, atol=1e-12)
        assert_allclose(ell["dop"], 1.0, atol=1e-12)                        # every ray is fully polarized

        # semi-axes and azimuth from the field over one period
        tau = numpy.linspace(0, 2 * numpy.pi, 7201)
        x = numpy.real(numpy.multiply.outer(Ex, numpy.exp(-1j * tau)))
        y = numpy.real(numpy.multiply.outer(Ey, numpy.exp(-1j * tau)))
        r = numpy.sqrt(x ** 2 + y ** 2)
        assert_allclose(ell["a"], r.max(axis=1), rtol=1e-6)
        assert_allclose(ell["b"], r.min(axis=1), atol=1e-3)                 # sampled minimum (b ~ 0 if linear)
        assert_allclose(ell["a"] ** 2 + ell["b"] ** 2, ell["S"][0], rtol=1e-12)
        assert_allclose(2 * ell["a"] * ell["b"], numpy.abs(ell["S"][3]), atol=1e-12)
        imax = r.argmax(axis=1)
        psi_field = numpy.mod(numpy.arctan2(y[numpy.arange(x.shape[0]), imax], x[numpy.arange(x.shape[0]), imax]),
                              numpy.pi)
        dpsi = numpy.abs(numpy.angle(numpy.exp(2j * (psi_field - ell["psi"])))) / 2  # modulo pi
        circular = ell["b"] > 0.99 * ell["a"]
        self.assertTrue(numpy.all(dpsi[~circular] < 1e-3))

        # sense of rotation of the field (sign of x dy/dt - y dx/dt) equals the sign of chi
        L = x[:, 0] * (y[:, 1] - y[:, 0]) - y[:, 0] * (x[:, 1] - x[:, 0])
        elliptical = numpy.abs(ell["chi"]) > 1e-6
        assert_allclose(numpy.sign(L[elliptical]), numpy.sign(ell["chi"][elliptical]))

    def test_reference_states(self):
        # SHADOW source: P=0.5 -> |Es|=|Ep|=2^-1/2; Phi = phase_p - phase_s
        a = numpy.full(1, 2 ** -0.5)
        z = numpy.zeros(1)
        cases = {   # Phi: (psi, chi) in degrees
            0.0:             (135, 0),     # field along u_s + u_p: psi = -45 deg (= 135 deg)
            numpy.pi:        (45, 0),      # field along u_s - u_p
            numpy.pi / 2:    (0, -45),     # right circular (RCP), psi undefined (set to 0)
            -numpy.pi / 2:   (0, 45),      # left circular (LCP)
        }
        for Phi, (psi, chi) in cases.items():
            ell = make_beam(a, a, z, z + Phi).get_polarization_ellipse(flag_return=2, iray=0, nolost=0)
            assert_allclose(numpy.degrees([ell["psi"], ell["chi"]]), [psi, chi], atol=1e-8, err_msg="Phi=%g" % Phi)
            assert_allclose([ell["a"] ** 2 + ell["b"] ** 2, ell["dop"]], [1, 1], atol=1e-12)

    def test_flag_return(self):
        b = random_beam()
        ell1 = b.get_polarization_ellipse(flag_return=1, nolost=0)
        ell0 = b.get_polarization_ellipse(flag_return=0, nolost=0)
        ell2 = b.get_polarization_ellipse(flag_return=2, iray=17, nolost=0)

        # average: Stokes parameters averaged over the rays, partially polarized
        assert_allclose(ell0["S"], ell1["S"].mean(axis=1), rtol=1e-12)
        self.assertLess(ell0["dop"], 1.0)
        assert_allclose(ell0["a"] ** 2 + ell0["b"] ** 2, ell0["dop"] * ell0["S"][0], rtol=1e-12)

        # single ray = row of all the rays
        for key in ["psi", "chi", "a", "b", "dop"]:
            assert_allclose(ell2[key], ell1[key][17], atol=1e-12, err_msg=key)
        assert_allclose(ell2["S"], ell1["S"][:, 17], atol=1e-12)
        assert_allclose(ell2["P"], ell1["P"][:, 17], atol=1e-12)

        with self.assertRaises(ValueError):
            b.get_polarization_ellipse(flag_return=3)

    def test_opposite_handedness(self):
        # two rays with opposite circular polarization: the average is unpolarized
        a = numpy.full(2, 2 ** -0.5)
        z = numpy.zeros(2)
        b = make_beam(a, a, z, numpy.array([numpy.pi / 2, -numpy.pi / 2]))
        ell = b.get_polarization_ellipse(flag_return=0, nolost=0)
        assert_allclose(ell["dop"], 0, atol=1e-12)
        assert_allclose([ell["a"], ell["b"]], [0, 0], atol=1e-7)          # sqrt of the rounding error of S

    def test_curve(self):
        b = random_beam(n=50)
        ell1 = b.get_polarization_ellipse(flag_return=1, nolost=0)
        X, Y = S4Beam.get_polarization_ellipse_curve(ell1, npoints=181)
        self.assertEqual(X.shape, (50, 181))

        ell2 = b.get_polarization_ellipse(flag_return=2, iray=3, nolost=0)
        x, y = S4Beam.get_polarization_ellipse_curve(ell2, npoints=181)
        self.assertEqual(x.shape, (181,))
        assert_allclose([x, y], [X[3], Y[3]], atol=1e-12)

        # the curve is the ellipse of the field: same semi-axes and same sense of rotation
        r = numpy.sqrt(X ** 2 + Y ** 2)
        assert_allclose(r.max(axis=1), ell1["a"], rtol=1e-12)
        assert_allclose(r.min(axis=1), ell1["b"], atol=1e-3)
        L = X[:, 0] * (Y[:, 1] - Y[:, 0]) - Y[:, 0] * (X[:, 1] - X[:, 0])
        elliptical = numpy.abs(ell1["chi"]) > 1e-6
        assert_allclose(numpy.sign(L[elliptical]), numpy.sign(ell1["chi"][elliptical]))


if __name__ == "__main__":
    unittest.main()
