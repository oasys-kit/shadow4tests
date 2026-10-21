__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Electric fields of the rays diffracted by a grating, with the Jones calculus (S4GratingElement and
# S4Beam.calculate_jones_scattering) with r_s = r_p = 1 (no grating efficiency): the complex electric field of each
# diffracted ray (in the frame of the moved grating) is compared with the vector form
#
#     E_out = (E_in . s) s + (E_in . p_in) p_out,
#
# with s = v_in x v_out / |v_in x v_out|, p_in = s x v_in, p_out = s x v_out (SHADOW convention).
# The fields must be perpendicular to the diffracted directions, and the intensities (and the Stokes parameters
# referred to the scattering plane) are unchanged. Plane VLS grating, orders +1 and -1, rotated around the beam and
# with movements (a rotation around the normal gives conical diffraction).
#
# Previously, the electric vectors of the incident rays were kept, so they were not perpendicular to the diffracted
# directions.
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.gratings.s4_plane_grating import S4PlaneGrating, S4PlaneGratingElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot

GRAZING = numpy.radians(4.0)


def get_source():
    # elliptically polarized beam (P=0.7, phase difference 0.6 rad), 1000 eV
    s = SourceGeometrical(nrays=1000, seed=123)
    s.set_spatial_type_rectangle(width=1e-4, height=1e-4)
    s.set_angular_distribution_uniform(hdiv1=-1e-3, hdiv2=1e-3, vdiv1=-1e-3, vdiv2=1e-3)
    s.set_energy_distribution_singleline(1000.0)
    s.set_polarization(polarization_degree=0.7, phase_diff=0.6, coherent_beam=1)
    return s.get_beam()


def efield(beam):
    """Complex electric vector of each ray."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    return j0[:, None] * e_S + j1[:, None] * e_P


def check_grating(beam, order, alpha, movements=None):
    """Returns the errors of the field, of its transversality (footprint and image) and of the intensity, and the
    maximum angle (rad) between the diffracted direction and the specular direction."""
    oe = S4PlaneGrating(name="grating", ruling=800e3, ruling_coeff_linear=1e5, ruling_coeff_quadratic=1e3,
                        f_ruling=1, order=order)
    element = S4PlaneGratingElement(optical_element=oe, movements=movements, input_beam=beam,
                                    coordinates=ElementCoordinates(p=1.0, q=1.0,
                                                                   angle_radial=numpy.pi / 2 - GRAZING,
                                                                   angle_radial_out=numpy.pi / 2 - GRAZING,
                                                                   angle_azimuthal=alpha))
    beam_image, footprint = element.trace_beam()

    # incident beam in the (moved) grating frame, and diffracted beam back in the moved grating frame
    beam_in = beam.duplicate()
    beam_in.rotate(alpha, axis=2)
    beam_in.rotate(GRAZING, axis=1)
    beam_out = footprint.duplicate()
    if movements is not None:
        beam_in.rot_for(OFFX=movements.offset_x, OFFY=movements.offset_y, OFFZ=movements.offset_z,
                        X_ROT=movements.rotation_x, Y_ROT=movements.rotation_y, Z_ROT=movements.rotation_z)
        beam_out.rot_for(X_ROT=movements.rotation_x, Y_ROT=movements.rotation_y, Z_ROT=movements.rotation_z)

    v_in = beam_in.get_columns([4, 5, 6]).T
    v_out = beam_out.get_columns([4, 5, 6]).T
    E_in = efield(beam_in)
    E_out = efield(beam_out)
    s = vector_norm(vector_cross(v_in, v_out))
    p_in = vector_cross(s, v_in)
    p_out = vector_cross(s, v_out)
    E_expected = vector_dot(E_in, s)[:, None] * s + vector_dot(E_in, p_in)[:, None] * p_out

    good = beam_out.get_column(10) > 0
    scale = numpy.abs(E_in).max()
    error = numpy.abs(E_out - E_expected)[good].max() / scale
    transversality = numpy.abs(vector_dot(E_out, v_out))[good].max() / scale
    v_image = beam_image.get_columns([4, 5, 6]).T
    transversality_image = numpy.abs(vector_dot(efield(beam_image), v_image))[good].max() / scale
    intensity = numpy.abs(beam_out.get_column(23) - beam_in.get_column(23))[good].max()
    v_specular = v_in.copy()
    v_specular[:, 2] *= -1
    if movements is None:
        deviation = numpy.arccos(numpy.clip(vector_dot(v_out, v_specular), -1, 1))[good].max()
    else:
        deviation = numpy.nan
    return error, transversality, transversality_image, intensity, deviation


class TestGratingJones(unittest.TestCase):

    def setUp(self):
        self.beam = get_source()

    def _check(self, order, alpha, movements=None):
        error, transversality, transversality_image, intensity, deviation = \
            check_grating(self.beam, order, alpha, movements=movements)
        assert_allclose(error, 0.0, atol=1e-12)
        assert_allclose(transversality, 0.0, atol=1e-12)
        assert_allclose(transversality_image, 0.0, atol=1e-12)
        assert_allclose(intensity, 0.0, atol=1e-12)
        if movements is None:
            self.assertGreater(deviation, 1e-3) # diffraction, not specular reflection

    def test_orders(self):
        for order in [1, -1]:
            self._check(order, alpha=0.0)

    def test_rotated(self):
        for order in [1, -1]:
            self._check(order, alpha=numpy.radians(30))

    def test_conical(self):
        # rotation around the grating normal: grooves not perpendicular to the plane of incidence
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4, rotation_x=1e-3, rotation_z=0.1)
        for order in [1, -1]:
            self._check(order, alpha=numpy.radians(-70), movements=movements)

    def test_evanescent_order(self):
        # 2 deg grazing: order +1 is evanescent (cos(beta) > 1), order -1 propagates.
        # The evanescent rays must be lost, without NaN in the beam.
        grazing = numpy.radians(2.0)
        for order, n_good in [(1, 0), (-1, self.beam.get_number_of_rays())]:
            oe = S4PlaneGrating(name="grating", ruling=800e3, f_ruling=1, order=order)
            element = S4PlaneGratingElement(optical_element=oe, input_beam=self.beam,
                                            coordinates=ElementCoordinates(p=1.0, q=1.0,
                                                                           angle_radial=numpy.pi / 2 - grazing,
                                                                           angle_radial_out=numpy.pi / 2 - grazing))
            beam_image, footprint = element.trace_beam()
            self.assertEqual((footprint.get_column(10) > 0).sum(), n_good)
            self.assertFalse(numpy.isnan(footprint.rays).any())
            self.assertFalse(numpy.isnan(beam_image.rays).any())


if __name__ == "__main__":
    unittest.main()
