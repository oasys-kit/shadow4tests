__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Electric fields of the rays refracted by an interface (S4InterfaceElement, used by lenses, CRLs and transfocators),
# with the Jones calculus (S4Beam.calculate_jones_scattering) without the Fresnel transmission amplitudes
# (t_s = t_p = 1): the complex electric field of each refracted ray (in the frame of the moved interface) is compared
# with the vector form
#
#     E_out = (E_in . s) s + (E_in . p_in) p_out,
#
# with s = v_in x v_out / |v_in x v_out|, p_in = s x v_in, p_out = s x v_out (SHADOW convention).
# The fields must be perpendicular to the refracted directions (previously, the electric vectors of the incident rays
# were kept). As the Jones matrix is the identity, the field is rotated with the ray: the sigma and pi directions
# of the incident ray are rotated (not replaced by those of the plane of incidence of each ray, which would scramble
# the Stokes parameters S1, S2 of the rays of a lens), and the Jones vector is unchanged. Also:
# - the movements of the interface are undone after the refraction (previously, only the forward movement was
#   applied, so the image frame moved with the interface): an interface between equal media, moved, gives the same
#   beam as the interface not moved;
# - the rays in total internal reflection are lost, without NaN.
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.refractors.s4_conic_interface import S4ConicInterface, S4ConicInterfaceElement
from shadow4.beamline.optical_elements.refractors.s4_lens import S4Lens, S4LensElement
from shadow4.beamline.optical_elements.refractors.s4_crl import S4CRL, S4CRLElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot

PLANE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
SPHERE = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2 * 0.5, 0.0] # radius 0.5 m


def get_source(divergence=0.3):
    # elliptically polarized beam (P=0.7, phase difference 0.6 rad), visible light (2 eV)
    s = SourceGeometrical(nrays=1000, seed=123)
    s.set_spatial_type_rectangle(width=1e-3, height=1e-3)
    s.set_angular_distribution_uniform(hdiv1=-divergence, hdiv2=divergence, vdiv1=-divergence, vdiv2=divergence)
    s.set_energy_distribution_singleline(2.0)
    s.set_polarization(polarization_degree=0.7, phase_diff=0.6, coherent_beam=1)
    return s.get_beam()


def efield(beam):
    """Complex electric vector of each ray."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    return j0[:, None] * e_S + j1[:, None] * e_P


def get_element(beam, n1, n2, conic_coefficients, alpha=0.0, movements=None):
    oe = S4ConicInterface(f_r_ind=0, r_ind_obj=n1, r_ind_ima=n2, conic_coefficients=conic_coefficients)
    return S4ConicInterfaceElement(optical_element=oe, movements=movements, input_beam=beam,
                                   coordinates=ElementCoordinates(p=0.1, q=0.1, angle_radial=0.0,
                                                                  angle_radial_out=numpy.pi, angle_azimuthal=alpha))


def check_interface(beam, n1, n2, conic_coefficients, alpha=0.0, movements=None):
    """Returns the errors of the field and of its transversality (footprint and image), the error of the intensity,
    and the maximum deviation angle (rad) of the refracted rays."""
    beam_image, footprint = get_element(beam, n1, n2, conic_coefficients, alpha, movements).trace_beam()

    # incident beam in the (moved) interface frame, and refracted beam back in the moved interface frame
    theta = numpy.pi / 2 # grazing angle of the coordinates (normal incidence)
    beam_in = beam.duplicate()
    beam_in.rotate(alpha, axis=2)
    beam_in.rotate(theta, axis=1)
    beam_in.translation([0.0, -0.1 * numpy.cos(theta), 0.1 * numpy.sin(theta)]) # p=0.1
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
    deviation = numpy.arccos(numpy.clip(vector_dot(v_in, v_out), -1, 1))[good].max()
    return error, transversality, transversality_image, intensity, deviation


class TestInterfaceJones(unittest.TestCase):

    def setUp(self):
        self.beam = get_source()

    def _check(self, n1, n2, conic_coefficients, alpha=0.0, movements=None):
        error, transversality, transversality_image, intensity, deviation = \
            check_interface(self.beam, n1, n2, conic_coefficients, alpha=alpha, movements=movements)
        assert_allclose(error, 0.0, atol=1e-12)
        assert_allclose(transversality, 0.0, atol=1e-12)
        assert_allclose(transversality_image, 0.0, atol=1e-12)
        assert_allclose(intensity, 0.0, atol=1e-12)
        self.assertGreater(deviation, 0.05) # large refraction angles

    def test_plane(self):
        self._check(1.0, 1.5, PLANE)
        self._check(1.5, 1.0, PLANE)

    def test_sphere_rotated_with_movements(self):
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4, offset_z=2e-4,
                                               rotation_x=0.05, rotation_y=0.1, rotation_z=0.03)
        self._check(1.0, 1.5, SPHERE, alpha=numpy.radians(30))
        self._check(1.0, 1.5, SPHERE, alpha=numpy.radians(-70), movements=movements)

    def test_movements_undone(self):
        # interface between equal media (no refraction): the moved interface gives the same beam as the one not moved
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-3, offset_y=2e-3, offset_z=1e-3,
                                               rotation_x=0.05, rotation_y=0.1, rotation_z=0.03)
        beam = get_source(1e-3)
        ref, _ = get_element(beam, 1.5, 1.5, SPHERE).trace_beam()
        out, _ = get_element(beam, 1.5, 1.5, SPHERE, movements=movements).trace_beam()
        for col in [1, 3, 4, 5, 6]:
            assert_allclose(out.get_column(col), ref.get_column(col), atol=1e-12, err_msg="column %d" % col)
        assert_allclose(efield(out), efield(ref), atol=1e-12)

    def test_total_internal_reflection(self):
        # from glass (n=1.5) to vacuum, critical angle 41.8 deg: the rays beyond it are lost, without NaN
        beam = get_source(1.0)
        beam_image, footprint = get_element(beam, 1.5, 1.0, PLANE).trace_beam()
        v = beam.get_columns([4, 5, 6]).T
        sin_theta1 = numpy.sqrt(1 - v[:, 1] ** 2)       # normal incidence: the normal is along y
        tir = sin_theta1 > 1 / 1.5
        self.assertTrue(tir.any() and (~tir).any())
        assert_allclose(footprint.get_column(10) > 0, ~tir)
        self.assertFalse(numpy.isnan(footprint.rays).any())
        self.assertFalse(numpy.isnan(beam_image.rays).any())


class TestLensJones(unittest.TestCase):

    def test_lens(self):
        # visible glass lens (two spherical interfaces): fields perpendicular to the directions, intensity conserved
        beam = get_source(0.05)
        oe = S4Lens(thickness=5e-3, surface_shape=1, radius=0.05, convex_to_the_beam=1,
                    ri_calculation_mode=0, refraction_index=1.5, attenuation_coefficient=0.0)
        element = S4LensElement(optical_element=oe, input_beam=beam,
                                coordinates=ElementCoordinates(p=0.1, q=0.1, angle_radial=0.0,
                                                               angle_radial_out=numpy.pi))
        beam_image, _ = element.trace_beam()
        good = beam_image.get_column(10) > 0
        self.assertGreater(good.sum(), 0.9 * good.size)
        v = beam_image.get_columns([4, 5, 6]).T
        self.assertFalse(numpy.isnan(beam_image.rays).any())
        assert_allclose(numpy.abs(vector_dot(efield(beam_image), v))[good].max(), 0.0, atol=1e-12)
        assert_allclose(beam_image.intensity(nolost=1), good.sum(), rtol=1e-12)

    def test_crl_xrays(self):
        # Be CRL at 10 keV (deviations of microradians): fields perpendicular to the directions, and the Jones
        # vectors and the Stokes parameters of the rays unchanged (no Fresnel amplitudes, no absorption)
        s = SourceGeometrical(nrays=1000, seed=7)
        s.set_spatial_type_gaussian(sigma_h=1e-4, sigma_v=1e-4)
        s.set_angular_distribution_gaussian(sigdix=1e-5, sigdiz=1e-5)
        s.set_energy_distribution_singleline(10000.0)
        s.set_polarization(polarization_degree=0.8, phase_diff=0.3, coherent_beam=1)
        beam = s.get_beam()
        oe = S4CRL(n_lens=10, piling_thickness=2.5e-3, surface_shape=2, convex_to_the_beam=0, cylinder_angle=0,
                   ri_calculation_mode=0, refraction_index=1 - 3.4e-6, attenuation_coefficient=0.0,
                   thickness=50e-6, radius=200e-6)
        element = S4CRLElement(optical_element=oe, input_beam=beam,
                               coordinates=ElementCoordinates(p=10.0, q=5.0, angle_radial=0.0,
                                                              angle_radial_out=numpy.pi))
        beam_image, _ = element.trace_beam()
        good = beam_image.get_column(10) > 0
        self.assertEqual(good.sum(), good.size)
        v = beam_image.get_columns([4, 5, 6]).T
        v0 = beam.get_columns([4, 5, 6]).T
        self.assertGreater(numpy.abs(v[:, 0] - v0[:, 0]).max(), 1e-6) # the lenses deviate the rays
        assert_allclose(numpy.abs(vector_dot(efield(beam_image), v)).max(), 0.0, atol=1e-12)
        for j_out, j_in in zip(beam_image.get_jones_components(), beam.get_jones_components()):
            assert_allclose(j_out, j_in, atol=1e-12)
        for col in [30, 31, 32, 33]:
            assert_allclose(beam_image.get_column(col), beam.get_column(col), atol=1e-12, err_msg="column %d" % col)


if __name__ == "__main__":
    unittest.main()
