#
# Spherical mirror (full sphere, tangential and sagittal cylinders) at grazing incidence, with the radius calculated
# from the focal distances p, q and the grazing angle (SurfaceCalculation.INTERNAL). The geometry is that of the
# former SHADOW3 comparison (p = q = 10 m, incidence angle 88.8 deg). A point source with divergence 1e-4 rad is
# imaged at q: the mirror focuses in the meridional (tangential cylinder) or sagittal (sagittal cylinder) plane, and
# the other direction is not focused (size = divergence * (p + q)).
#
import unittest
import numpy
from syned.beamline.shape import Convexity, Direction
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_sphere_mirror import S4SphereMirror, S4SphereMirrorElement
from shadow4tests.test_beamline.mirror_test_tools import source_beam, trace_mirror, columns

P, Q, THETA = 10.0, 10.0, numpy.radians(90 - 88.8)
DIV = 1e-4


def sphere(is_cylinder=False, cylinder_direction=Direction.TANGENTIAL):
    return S4SphereMirror(surface_calculation=SurfaceCalculation.INTERNAL, is_cylinder=is_cylinder,
                          cylinder_direction=cylinder_direction, convexity=Convexity.DOWNWARD,
                          p_focus=P, q_focus=Q, grazing_angle=THETA)


class TestSphereMirror(unittest.TestCase):

    def test_radius(self):
        # meridional radius R = 2 p q / ((p + q) sin(theta)); surface x^2 + y^2 + z^2 - 2 R z = 0
        R = 2 * P * Q / ((P + Q) * numpy.sin(THETA))
        ccc = sphere().get_optical_surface_instance().ccc
        numpy.testing.assert_allclose(ccc, [1, 1, 1, 0, 0, 0, 0, 0, -2 * R, 0], rtol=1e-6, atol=1e-9)
        # sagittal cylinder: radius rho = 2 p q sin(theta) / (p + q)
        rho = 2 * P * Q * numpy.sin(THETA) / (P + Q)
        ccc = sphere(True, Direction.SAGITTAL).get_optical_surface_instance().ccc
        self.assertAlmostEqual(ccc[8], -2 * rho, places=6)

    def test_sphere(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4SphereMirrorElement, sphere(), P, Q, THETA, source_beam(divergence=DIV)))
        self.assertLess(z.std(), 1e-6)                                     # meridional focus (aberrations only)
        self.assertAlmostEqual(x.std() / (DIV * (P + Q)), 1.0, delta=0.03)  # not focused sagittally

    def test_tangential_cylinder(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4SphereMirrorElement, sphere(True, Direction.TANGENTIAL), P, Q, THETA,
                                                source_beam(divergence=DIV)))
        self.assertLess(z.std(), 1e-6)
        self.assertAlmostEqual(x.std() / (DIV * (P + Q)), 1.0, delta=0.03)

    def test_sagittal_cylinder(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4SphereMirrorElement, sphere(True, Direction.SAGITTAL), P, Q, THETA,
                                                source_beam(divergence=DIV)))
        self.assertLess(x.std(), 2e-5)                                     # sagittal focus (aberrations)
        self.assertAlmostEqual(z.std() / (DIV * (P + Q)), 1.0, delta=0.03)  # not focused meridionally


if __name__ == "__main__":
    unittest.main()
