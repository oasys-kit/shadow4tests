__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Mirror reflectivity with the Jones calculus for a curved mirror (sphere), rotated around the beam (angle_azimuthal)
# and with movements: the complex electric field of each reflected ray (in the frame of the moved mirror) is compared
# with the direct vector form of the Fresnel equations:
#
#     E_out = r_s (E_in . s) s + r_p (E_in . p_in) p_out,
#
# with s = v_in x v_out / |v_in x v_out|, p_in = s x v_in, p_out = s x v_out (SHADOW convention), and r_s, r_p
# (Born & Wolf) calculated at the local grazing angle of each ray (the normal is along v_out - v_in).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_sphere_mirror import S4SphereMirror, S4SphereMirrorElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.physical_models.prerefl.prerefl import PreRefl
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot

N_GLASS = 1.51 + 0j            # glass, visible light (7.5 eV)
GRAZING = numpy.radians(35)    # large divergence: local grazing angles from ~0.5 to ~60 deg (includes Brewster, 33.5 deg)


def get_source():
    # elliptically polarized beam (P=0.7, phase difference 0.6 rad), large divergence in both directions
    s = SourceGeometrical(nrays=2000, seed=123)
    s.set_spatial_type_rectangle(width=1e-3, height=1e-3)
    s.set_angular_distribution_uniform(hdiv1=-0.2, hdiv2=0.2, vdiv1=-0.2, vdiv2=0.2)
    s.set_energy_distribution_singleline(7.5, unit='eV')
    s.set_polarization(polarization_degree=0.7, phase_diff=0.6, coherent_beam=1)
    return s.get_beam()


def efield(beam):
    """Complex electric vector of each ray."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    return j0[:, None] * e_S + j1[:, None] * e_P


def check_curved_mirror(beam, alpha, movements=None):
    """Returns max |E - E_expected| / max|E_in|, max |E . v_out| and the range of local grazing angles (deg)."""
    oe = S4SphereMirror(surface_calculation=0, is_cylinder=0, p_focus=1.0, q_focus=1.0, grazing_angle=GRAZING,
                        f_reflec=1, f_refl=1, refraction_index=N_GLASS)
    element = S4SphereMirrorElement(optical_element=oe, movements=movements, input_beam=beam,
                                    coordinates=ElementCoordinates(p=1.0, q=1.0, angle_radial=numpy.pi / 2 - GRAZING,
                                                                   angle_azimuthal=alpha))
    _, footprint = element.trace_beam()

    # incident beam in the (moved) mirror frame, and reflected beam back in the moved mirror frame
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

    normal = vector_norm(v_out - v_in)
    grazing = numpy.arcsin(numpy.abs(vector_dot(v_in, normal)))
    rs, rp = PreRefl.reflectivity_amplitudes_fresnel_external(photon_energy_ev=7.5,
                                                               refraction_index_1=1.0 + 0j,
                                                               refraction_index_2=N_GLASS,
                                                               grazing_angle_mrad=grazing * 1e3,
                                                               method=0)
    s = vector_norm(vector_cross(v_in, v_out))
    p_in = vector_cross(s, v_in)
    p_out = vector_cross(s, v_out)
    E_expected = (rs * vector_dot(E_in, s))[:, None] * s + (rp * vector_dot(E_in, p_in))[:, None] * p_out

    good = beam_out.get_column(10) > 0
    error = numpy.abs(E_out - E_expected)[good].max() / numpy.abs(E_in).max()
    transversality = numpy.abs(vector_dot(E_out, v_out))[good].max()
    return error, transversality, numpy.degrees(grazing[good].min()), numpy.degrees(grazing[good].max())


class TestMirrorJonesCurved(unittest.TestCase):

    def setUp(self):
        self.beam = get_source()

    def _check(self, alpha, movements=None):
        error, transversality, gmin, gmax = check_curved_mirror(self.beam, alpha, movements=movements)
        self.assertLess(gmin, 2.0)     # the local grazing angles go from grazing incidence
        self.assertGreater(gmax, 50.0) # to beyond Brewster (grazing 33.5 deg for n=1.51)
        assert_allclose(error, 0.0, atol=1e-12)
        assert_allclose(transversality, 0.0, atol=1e-12)

    def test_sphere(self):
        self._check(alpha=0.0)

    def test_sphere_rotated(self):
        self._check(alpha=numpy.radians(30))

    def test_sphere_rotated_with_movements(self):
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4,
                                               rotation_x=0.05, rotation_y=0.1, rotation_z=0.03)
        self._check(alpha=numpy.radians(-70), movements=movements)


if __name__ == "__main__":
    unittest.main()
