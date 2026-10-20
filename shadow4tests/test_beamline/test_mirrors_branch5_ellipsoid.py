#
# Ellipsoidal mirror (ellipsoid of revolution and tangential elliptical cylinder) with foci at p and q
# (SurfaceCalculation.INTERNAL), in the geometry of the former SHADOW3 comparison (p = q = 10 m, incidence angle
# 88.8 deg). A point source at the first focus is imaged without aberrations at the second focus.
#
import unittest
import numpy
from syned.beamline.shape import Convexity, Direction
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror, S4EllipsoidMirrorElement
from shadow4tests.test_beamline.mirror_test_tools import source_beam, trace_mirror, columns

P, Q, THETA = 10.0, 10.0, numpy.radians(90 - 88.8)
DIV = 1e-4


def ellipsoid(is_cylinder=False, p=P, q=Q):
    return S4EllipsoidMirror(surface_calculation=SurfaceCalculation.INTERNAL, is_cylinder=is_cylinder,
                             cylinder_direction=Direction.TANGENTIAL, convexity=Convexity.DOWNWARD,
                             p_focus=p, q_focus=q, grazing_angle=THETA)


class TestEllipsoidMirror(unittest.TestCase):

    def test_ellipsoid(self):
        for p, q in [(P, Q), (20.0, 5.0)]:
            x, z, vx, vy, vz = columns(trace_mirror(S4EllipsoidMirrorElement, ellipsoid(p=p, q=q), p, q, THETA,
                                                    source_beam(divergence=DIV)))
            self.assertLess(x.std(), 1e-9, msg="p=%g q=%g" % (p, q))       # point-to-point imaging
            self.assertLess(z.std(), 1e-9, msg="p=%g q=%g" % (p, q))
            self.assertAlmostEqual(vz.std() / (DIV * p / q), 1.0, delta=0.03)  # angular magnification p/q

    def test_tangential_cylinder(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4EllipsoidMirrorElement, ellipsoid(True), P, Q, THETA,
                                                source_beam(divergence=DIV)))
        self.assertLess(z.std(), 1e-9)                                     # meridional focus
        self.assertAlmostEqual(x.std() / (DIV * (P + Q)), 1.0, delta=0.03)  # not focused sagittally

    def test_source_size(self):
        # extended source (5 um, as in the former test): unit magnification for p = q
        x, z, vx, vy, vz = columns(trace_mirror(S4EllipsoidMirrorElement, ellipsoid(), P, Q, THETA,
                                                source_beam(sigma=5e-6, divergence=1e-6)))
        self.assertAlmostEqual(z.std() / 5e-6, 1.0, delta=0.05)
        self.assertAlmostEqual(x.std() / 5e-6, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
