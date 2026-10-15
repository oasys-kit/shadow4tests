__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# The python scripts exported by S4BeamlineElement.to_python_code() must reproduce the coordinates and movements
# exactly. With %.10g, pi was written as 3.141592654 (error 4e-10 rad), which is enough to destroy the extinction
# of a crystal polarizer/analyzer pair in the exported script.
#

import unittest
import numpy

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement


class TestPythonCodePrecision(unittest.TestCase):

    def test_coordinates_and_movements_round_trip(self):
        values = dict(p=numpy.float64(10.0) / 3, q=1.0 / 7,
                      angle_radial=numpy.radians(44.123456789123), angle_azimuthal=numpy.pi,
                      angle_radial_out=numpy.pi / 2 - 1e-13)
        moves = dict(f_move=1, offset_x=1.234567891234e-7, offset_y=-2.0 / 3, offset_z=0.0,
                     rotation_x=numpy.radians(0.1234567891234), rotation_y=numpy.pi / 4, rotation_z=-1e-12)

        element = S4PlaneMirrorElement(optical_element=S4PlaneMirror(),
                                       coordinates=ElementCoordinates(**values),
                                       movements=S4BeamlineElementMovements(**moves))

        namespace = {}
        exec(element.to_python_code_coordinates() + element.to_python_code_movements(), namespace)
        c = namespace["coordinates"]
        m = namespace["movements"]

        self.assertEqual(c.p(), values["p"])
        self.assertEqual(c.q(), values["q"])
        self.assertEqual(c.angle_radial(), values["angle_radial"])
        self.assertEqual(c.angle_azimuthal(), values["angle_azimuthal"])
        self.assertEqual(c.angle_radial_out(), values["angle_radial_out"])
        for key, value in moves.items():
            self.assertEqual(getattr(m, key), value, msg=key)

        self.assertNotIn("np.float64", element.to_python_code_coordinates())


if __name__ == "__main__":
    unittest.main()
