__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Crystal diffraction with the Jones calculus, full trace (S4PlaneCrystalElement.trace_beam): the complex electric
# field of each diffracted ray (in the frame of the moved crystal) is compared with the vector form
#
#     E_out = r_s (E_in . s) s + r_p (E_in . p_in) p_out,
#
# with s = v_in x v_out / |v_in x v_out|, p_in = s x v_in, p_out = s x v_out (SHADOW convention), and r_s, r_p the
# crystalpy amplitudes. Note that p_in . p_out = v_in . v_out = cos(2 theta_B), the sign of the polarization factor
# used by crystalpy for the pi polarization. Si 111 (theta_B < 45 deg) and Si 333 (theta_B > 45 deg) at 8 keV,
# crystal rotated around the beam (angle_azimuthal) and with movements. The fields obtained with the standard Jones
# algebra (method_efields_management=2) must be the same.
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal, S4PlaneCrystalElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot

ENERGY = 8000.0


def get_source(divergence):
    # elliptically polarized beam (P=0.7, phase difference 0.6 rad), divergence covering the Darwin curve
    s = SourceGeometrical(nrays=1000, seed=123)
    s.set_spatial_type_rectangle(width=1e-4, height=1e-4)
    s.set_angular_distribution_uniform(hdiv1=-divergence, hdiv2=divergence, vdiv1=-divergence, vdiv2=divergence)
    s.set_energy_distribution_singleline(ENERGY)
    s.set_polarization(polarization_degree=0.7, phase_diff=0.6, coherent_beam=1)
    return s.get_beam()


def efield(beam):
    """Complex electric vector of each ray."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    return j0[:, None] * e_S + j1[:, None] * e_P


def get_element(beam, hkl, alpha, movements, method_efields_management):
    oe = S4PlaneCrystal(name="xtal", material="Si", miller_index_h=hkl[0], miller_index_k=hkl[1],
                        miller_index_l=hkl[2], is_thick=1, f_central=1, f_phot_cent=0, phot_cent=ENERGY,
                        material_constants_library_flag=1, method_efields_management=method_efields_management)
    return S4PlaneCrystalElement(optical_element=oe, movements=movements, input_beam=beam,
                                 coordinates=ElementCoordinates(p=1.0, q=1.0, angle_radial=0.0,
                                                                angle_azimuthal=alpha, angle_radial_out=0.0))


def check_crystal(beam, hkl, alpha, movements=None):
    """Returns the errors of the field and of its transversality, the difference between methods 0 and 2,
    the mean reflectivities (sigma, pi) and the Bragg angle (deg)."""
    element = get_element(beam, hkl, alpha, movements, 0)
    _, footprint = element.trace_beam()
    _, footprint2 = get_element(beam, hkl, alpha, movements, 2).trace_beam()

    # incident beam in the (moved) crystal frame, and diffracted beam back in the moved crystal frame
    theta = numpy.pi / 2 - element.get_coordinates().angle_radial()
    beam_in = beam.duplicate()
    beam_in.rotate(alpha, axis=2)
    beam_in.rotate(theta, axis=1)
    beam_in.translation([0.0, -numpy.cos(theta), numpy.sin(theta)]) # p=1
    beam_out = footprint.duplicate()
    if movements is not None:
        beam_in.rot_for(OFFX=movements.offset_x, OFFY=movements.offset_y, OFFZ=movements.offset_z,
                        X_ROT=movements.rotation_x, Y_ROT=movements.rotation_y, Z_ROT=movements.rotation_z)
        beam_out.rot_for(X_ROT=movements.rotation_x, Y_ROT=movements.rotation_y, Z_ROT=movements.rotation_z)

    # crystalpy directions and amplitudes for the incident rays
    intercept, normal = element.get_optical_element().get_optical_surface_instance().calculate_intercept_on_beam(beam_in)
    v_in, v_out, rs, rp = element._calculate_perfect_crystal_scattering(intercept, normal)
    assert_allclose(beam_out.get_columns([4, 5, 6]).T, v_out, atol=1e-12)

    E_in = efield(beam_in)
    E_out = efield(beam_out)
    s = vector_norm(vector_cross(v_in, v_out))
    p_in = vector_cross(s, v_in)
    p_out = vector_cross(s, v_out)
    E_expected = (rs * vector_dot(E_in, s))[:, None] * s + (rp * vector_dot(E_in, p_in))[:, None] * p_out

    good = beam_out.get_column(10) > 0
    scale = numpy.abs(E_in).max()
    error = numpy.abs(E_out - E_expected)[good].max() / scale
    transversality = numpy.abs(vector_dot(E_out, v_out))[good].max() / scale
    methods = numpy.abs(efield(footprint) - efield(footprint2)).max() / scale
    return error, transversality, methods, (numpy.abs(rs) ** 2).mean(), (numpy.abs(rp) ** 2).mean(), \
           numpy.degrees(theta)


class TestCrystalJonesTrace(unittest.TestCase):

    def _check(self, hkl, divergence, alpha, movements=None):
        beam = get_source(divergence)
        error, transversality, methods, Rs, Rp, theta = check_crystal(beam, hkl, alpha, movements=movements)
        self.assertGreater(Rs, 0.1) # the rays are in the Darwin curve
        assert_allclose(error, 0.0, atol=1e-12)
        assert_allclose(transversality, 0.0, atol=1e-12)
        assert_allclose(methods, 0.0, atol=1e-12)
        return Rs, Rp, theta

    def test_si111(self):
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4, rotation_x=5e-6, rotation_z=0.05)
        for alpha, mov in [(0.0, None), (numpy.radians(30), None), (numpy.radians(-70), movements)]:
            Rs, Rp, theta = self._check((1, 1, 1), 60e-6, alpha, mov)
            self.assertLess(theta, 45.0)
            self.assertGreater(Rp, 0.5 * Rs)  # cos(2 theta_B) = 0.88

    def test_si333(self):
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4, rotation_x=1e-6, rotation_z=0.05)
        for alpha, mov in [(0.0, None), (numpy.radians(30), None), (numpy.radians(-70), movements)]:
            Rs, Rp, theta = self._check((3, 3, 3), 15e-6, alpha, mov)
            self.assertGreater(theta, 45.0)
            self.assertLess(Rp, 0.1 * Rs)     # cos(2 theta_B) = -0.10


if __name__ == "__main__":
    unittest.main()
