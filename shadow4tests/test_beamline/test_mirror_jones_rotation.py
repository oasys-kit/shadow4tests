__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Mirror reflectivity with the Jones calculus (S4MirrorElement.trace_beam + S4Beam.calculate_jones_scattering):
# the incident field is projected onto the sigma/pi directions of the scattering plane of each ray, multiplied by
# (rs, rp), and expressed along the sigma/pi directions of the reflected ray.
#
# Previously, rs and rp were applied to the sigma and pi components of the beam (wrong for a mirror rotated around
# the beam), and the electric vectors were not perpendicular to the reflected direction.
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
from shadow4.physical_models.prerefl.prerefl import PreRefl
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot

N_GLASS = 1.51 + 0j      # glass, visible light (7.5 eV)


def get_source(P, nrays=500):
    # vertical divergence: scan of the incidence angle on a mirror at alpha=0
    s = SourceGeometrical(nrays=nrays, seed=5676561)
    s.set_spatial_type_point()
    s.set_depth_distribution_off()
    s.set_angular_distribution_uniform(hdiv1=-1e-10, hdiv2=1e-10, vdiv1=-0.8, vdiv2=0.8)
    s.set_energy_distribution_singleline(7.5, unit='eV')
    s.set_polarization(polarization_degree=P, phase_diff=0, coherent_beam=1)
    return s.get_beam()


def trace_mirror(beam, alpha):
    oe = S4PlaneMirror(name='glass', f_reflec=1, f_refl=1, refraction_index=N_GLASS)
    coordinates = ElementCoordinates(p=0.1, q=0.1, angle_radial=numpy.pi / 4, angle_azimuthal=alpha,
                                     angle_radial_out=numpy.pi / 4)
    return S4PlaneMirrorElement(optical_element=oe, coordinates=coordinates, input_beam=beam).trace_beam()


def rebase(beam):
    """Same physical field, stored in the default basis (sigma = projection of x, pi = sigma x v)."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    E = j0[:, None] * e_S + j1[:, None] * e_P
    v = beam.get_columns([4, 5, 6]).T
    s = numpy.zeros_like(v)
    s[:, 0] = 1.0
    s = vector_norm(s - vector_dot(s, v)[:, None] * v)
    p = vector_norm(vector_cross(s, v))
    beam.set_jones_components(vector_dot(E, s), vector_dot(E, p), e_S=s, e_P=p)


class TestMirrorJones(unittest.TestCase):

    def test_rotated_mirror_equivalence(self):
        # case 1: field along u_s + u_p (P=0.5), divergence in the plane of a mirror at alpha=0
        b1 = get_source(0.5)
        _, f1 = trace_mirror(b1, 0.0)
        # case 2: the same beam rotated by 45 deg around y (horizontal field, divergence plane at 45 deg), with the
        # field stored in the default (horizontal) sigma basis, and the mirror rotated to alpha=-45 deg
        b2 = b1.duplicate()
        b2.rotate(numpy.pi / 4, axis=2)
        rebase(b2)
        self.assertGreater(b2.get_column(24).sum(), 0.95 * b2.get_column(23).sum())   # field (mostly) along sigma
        _, f2 = trace_mirror(b2, -numpy.pi / 4)
        assert_allclose(f2.get_column(24), f1.get_column(24), atol=1e-12)              # s intensity
        assert_allclose(f2.get_column(25), f1.get_column(25), atol=1e-12)              # p intensity
        dphase = numpy.angle(numpy.exp(1j * (f2.get_column(39) - f1.get_column(39))))
        assert_allclose(dphase, 0.0, atol=1e-9)                                         # phase s - p

    def test_transverse_fields(self):
        _, footprint = trace_mirror(get_source(0.3), numpy.radians(30.0))
        v = footprint.get_columns([4, 5, 6]).T
        Es = footprint.get_columns([7, 8, 9]).T
        Ep = footprint.get_columns([16, 17, 18]).T
        self.assertLess(numpy.abs(vector_dot(Es, v)).max(), 1e-12)
        self.assertLess(numpy.abs(vector_dot(Ep, v)).max(), 1e-12)
        self.assertLess(numpy.abs(vector_dot(Es, Ep)).max(), 1e-12)

    def test_alpha_zero_fresnel(self):
        # mirror at alpha=0: s and p intensities are |rs|^2 Is and |rp|^2 Ip of the incident rays
        beam = get_source(0.7)
        _, footprint = trace_mirror(beam, 0.0)
        theta_g = numpy.abs(numpy.arcsin(footprint.get_column(6)))          # grazing angle (plane mirror)
        rs, rp = PreRefl.reflectivity_amplitudes_fresnel_external(photon_energy_ev=7.5,
                                                                  refraction_index_1=numpy.ones(theta_g.size, dtype=complex),
                                                                  refraction_index_2=numpy.full(theta_g.size, N_GLASS),
                                                                  grazing_angle_mrad=1e3 * theta_g,
                                                                  roughness_rms_A=0.0, method=0)
        assert_allclose(footprint.get_column(24), numpy.abs(rs) ** 2 * beam.get_column(24), rtol=1e-6, atol=1e-12)
        assert_allclose(footprint.get_column(25), numpy.abs(rp) ** 2 * beam.get_column(25), rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
