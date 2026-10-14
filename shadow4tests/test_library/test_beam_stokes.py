__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Stokes columns (30-33, 42-44) must follow the conventions of the Jones/Mueller-Stokes draft
# ("Producing and Measuring Circularly Polarized X-rays", unpublished): computed in the right-handed
# basis (u_s, -u_p, v), i.e. Ex=Es, Ey=-Ep. Columns 40/41 are the complex amplitudes |E| exp(i phi).
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


class TestBeamStokes(unittest.TestCase):

    def test_stokes_columns(self):
        rng = numpy.random.default_rng(7)
        n = 10
        a_s, a_p = rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n)
        phs, php = rng.uniform(0, 2 * numpy.pi, n), rng.uniform(0, 2 * numpy.pi, n)
        b = make_beam(a_s, a_p, phs, php)

        Ex = a_s * numpy.exp(1j * phs)
        Ey = -a_p * numpy.exp(1j * php)
        S = [abs(Ex) ** 2 + abs(Ey) ** 2,
             abs(Ex) ** 2 - abs(Ey) ** 2,
             2 * numpy.real(Ex * numpy.conj(Ey)),
             2 * numpy.imag(numpy.conj(Ex) * Ey)]
        for k, col in enumerate([30, 31, 32, 33]):
            assert_allclose(b.get_column(col), S[k], atol=1e-12, err_msg="column %d" % col)
        for k, col in enumerate([42, 43, 44]):
            assert_allclose(b.get_column(col), S[k + 1] / S[0], atol=1e-12, err_msg="column %d" % col)

    def test_reference_states(self):
        # SHADOW source: P=0.5 -> |Es|=|Ep|=2^-1/2; Phi = phase_p - phase_s
        a = numpy.full(1, 2 ** -0.5)
        z = numpy.zeros(1)
        cases = {   # Phi: (S1, S2, S3)
            0.0:             (0, -1, 0),   # field along u_s + u_p: -45 deg for an observer looking toward the source
            numpy.pi:        (0, 1, 0),
            numpy.pi / 2:    (0, 0, -1),   # right circular (RCP)
            -numpy.pi / 2:   (0, 0, 1),    # left circular (LCP)
        }
        for Phi, expected in cases.items():
            b = make_beam(a, a, z, z + Phi)
            got = [b.get_column(c)[0] for c in [42, 43, 44]]
            assert_allclose(got, expected, atol=1e-12, err_msg="Phi=%g" % Phi)

    def test_complex_amplitudes(self):
        a_s, a_p = numpy.array([0.6, 0.3]), numpy.array([0.8, 0.2])
        phs, php = numpy.array([0.1, 2.0]), numpy.array([-1.0, 0.5])
        b = make_beam(a_s, a_p, phs, php)
        assert_allclose(b.get_column(40), a_s * numpy.exp(1j * phs), atol=1e-12)
        assert_allclose(b.get_column(41), a_p * numpy.exp(1j * php), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
