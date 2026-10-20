#
# Mirror defined by its conic coefficients (S4ConicMirror). The coefficients of the former SHADOW3 comparison,
# x^2 + y^2 + z^2 - 954.999 z = 0, are a sphere of radius 477.5 m, i.e., the sphere that focuses at p = q = 10 m with
# incidence angle 88.8 deg. The conic mirror must give the same beam as the spherical mirror.
#
import unittest
import numpy
from syned.beamline.shape import Convexity
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror, S4ConicMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_sphere_mirror import S4SphereMirror, S4SphereMirrorElement
from shadow4tests.test_beamline.mirror_test_tools import source_beam, trace_mirror, columns

P, Q, THETA = 10.0, 10.0, numpy.radians(90 - 88.8)
CCC = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -954.999, 0.0]


class TestConicCoefficientsMirror(unittest.TestCase):

    def test_conic_equals_sphere(self):
        beam = source_beam(divergence=1e-4)
        b_conic = trace_mirror(S4ConicMirrorElement, S4ConicMirror(conic_coefficients=CCC), P, Q, THETA, beam)
        sphere = S4SphereMirror(surface_calculation=SurfaceCalculation.INTERNAL, convexity=Convexity.DOWNWARD,
                                p_focus=P, q_focus=Q, grazing_angle=THETA)
        b_sphere = trace_mirror(S4SphereMirrorElement, sphere, P, Q, THETA, beam)
        for c, cs in zip(columns(b_conic), columns(b_sphere)):
            numpy.testing.assert_allclose(c, cs, atol=1e-8)
        x, z, vx, vy, vz = columns(b_conic)
        self.assertLess(z.std(), 1e-6)                        # meridional focus at q


if __name__ == "__main__":
    unittest.main()
