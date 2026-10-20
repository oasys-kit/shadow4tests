#
# Paraboloidal mirror (paraboloid of revolution and tangential parabolic cylinder) (SurfaceCalculation.INTERNAL), in
# the geometry of the former SHADOW3 comparison (p = 20 m, q = 10 m, incidence angle 89.828 deg, i.e., 3 mrad
# grazing). Collimating paraboloid (image at infinity): a point source at the focus gives a parallel beam. Focusing
# paraboloid (source at infinity): a parallel beam is focused at q.
#
import unittest
import numpy
from syned.beamline.shape import Convexity, Direction, Side
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_paraboloid_mirror import S4ParaboloidMirror, S4ParaboloidMirrorElement
from shadow4tests.test_beamline.mirror_test_tools import source_beam, trace_mirror, columns

P, Q, THETA = 20.0, 10.0, numpy.radians(90 - 89.8281126615)
DIV = 1e-4


def paraboloid(at_infinity, is_cylinder=False):
    return S4ParaboloidMirror(surface_calculation=SurfaceCalculation.INTERNAL, at_infinity=at_infinity,
                              is_cylinder=is_cylinder, cylinder_direction=Direction.TANGENTIAL,
                              convexity=Convexity.DOWNWARD, p_focus=P, q_focus=Q, grazing_angle=THETA)


class TestParaboloidMirror(unittest.TestCase):

    def test_collimating(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4ParaboloidMirrorElement, paraboloid(Side.IMAGE), P, Q, THETA,
                                                source_beam(divergence=DIV)))
        self.assertLess(vx.std(), 1e-9)                                    # parallel output beam
        self.assertLess(vz.std(), 1e-9)

    def test_collimating_tangential_cylinder(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4ParaboloidMirrorElement, paraboloid(Side.IMAGE, True), P, Q, THETA,
                                                source_beam(divergence=DIV)))
        self.assertLess(vz.std(), 1e-9)                                    # collimated in the meridional plane
        self.assertAlmostEqual(vx.std() / DIV, 1.0, delta=0.03)            # unchanged in the sagittal plane

    def test_focusing(self):
        x, z, vx, vy, vz = columns(trace_mirror(S4ParaboloidMirrorElement, paraboloid(Side.SOURCE), P, Q, THETA,
                                                source_beam(sigma=5e-6, divergence=0.0)))
        self.assertLess(x.std(), 1e-9)                                     # parallel beam focused at q
        self.assertLess(z.std(), 1e-9)


if __name__ == "__main__":
    unittest.main()
