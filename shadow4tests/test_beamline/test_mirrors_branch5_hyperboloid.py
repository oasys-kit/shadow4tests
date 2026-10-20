#
# Hyperboloidal mirror (SurfaceCalculation.INTERNAL) with foci at p = 10 m and q = 1000 m and incidence angle
# 88.8 deg (geometry of the former SHADOW3 comparison). A point source at the first focus gives, after reflection,
# rays that diverge from the second (virtual) focus, at a distance q behind the mirror: the angular magnification is
# p/q.
#
import unittest
import numpy
from syned.beamline.shape import Convexity
from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
from shadow4.beamline.optical_elements.mirrors.s4_hyperboloid_mirror import S4HyperboloidMirror, S4HyperboloidMirrorElement
from shadow4tests.test_beamline.mirror_test_tools import source_beam, trace_mirror, columns

P, Q, THETA = 10.0, 1000.0, numpy.radians(90 - 88.8)
DIV = 1e-4
Q_IMAGE = 10.0   # distance from the mirror to the image plane


class TestHyperboloidMirror(unittest.TestCase):

    def test_virtual_focus(self):
        oe = S4HyperboloidMirror(surface_calculation=SurfaceCalculation.INTERNAL, convexity=Convexity.DOWNWARD,
                                 p_focus=P, q_focus=Q, grazing_angle=THETA)
        x, z, vx, vy, vz = columns(trace_mirror(S4HyperboloidMirrorElement, oe, P, Q_IMAGE, THETA,
                                                source_beam(divergence=DIV)))
        D = -(Q + Q_IMAGE)                         # back-propagation from the image plane to the virtual focus
        self.assertLess((x + vx / vy * D).std(), 1e-9)
        self.assertLess((z + vz / vy * D).std(), 1e-9)
        self.assertAlmostEqual(vz.std() / (DIV * P / Q), 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
