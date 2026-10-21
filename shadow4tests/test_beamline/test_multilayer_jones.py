__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Multilayer reflectivity with the Jones calculus (S4MultilayerElement.trace_beam + S4Beam.calculate_jones_scattering).
#
# 1) Conventions of MLayer: it uses n = 1 - delta - i beta (time factor exp(+i omega t)); its amplitudes are the
#    complex conjugates of those in the SHADOW4 convention (exp(-i omega t)), with the sign of rp of Born & Wolf.
#    Checked against an independent Parratt recursion (exp(-i omega t)) for a stack, and against the Fresnel
#    equations (PreRefl) for the bare substrate.
# 2) Full trace: the complex electric field of each reflected ray (in the frame of the moved multilayer) is compared
#    with the vector form
#
#        E_out = r_s (E_in . s) s + r_p (E_in . p_in) p_out,
#
#    with s = v_in x v_out / |v_in x v_out|, p_in = s x v_in, p_out = s x v_out (SHADOW convention) and
#    r_s = conj(r_s,MLayer), r_p = conj(r_p,MLayer) at the local grazing angle of each ray. Hard X-ray multilayer
#    (B/W at 8 keV) rotated around the beam and with movements, and a soft X-ray polarizer (Bragg angle 45 deg).
#
# Previously, the phases of the amplitudes were not applied, and the moduli were applied to the sigma and pi
# components of the beam (wrong for a multilayer rotated around the beam).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.multilayers.s4_plane_multilayer import S4PlaneMultilayer, \
    S4PlaneMultilayerElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.physical_models.mlayer.mlayer import MLayer
from shadow4.physical_models.prerefl.prerefl import PreRefl
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot


def parratt(grazing, wavelength, layers, n_substrate):
    """Reflection amplitudes (s, p; sign of Born & Wolf for p) of a stack, time factor exp(-i omega t).
    layers: list of (n, thickness) from top to bottom (thickness in the units of wavelength)."""
    c2 = numpy.cos(grazing) ** 2
    n = [1.0 + 0j] + [layer[0] for layer in layers] + [n_substrate]
    d = [0.0] + [layer[1] for layer in layers] + [0.0]
    kz = [numpy.sqrt(nj ** 2 - c2 + 0j) for nj in n]
    out = []
    for pol in "sp":
        R = 0.0
        for j in range(len(n) - 2, -1, -1):
            a, b = kz[j], kz[j + 1]
            if pol == "s":
                r = (a - b) / (a + b)
            else:
                r = (n[j + 1] ** 2 * a - n[j] ** 2 * b) / (n[j + 1] ** 2 * a + n[j] ** 2 * b)
            phase = numpy.exp(2j * 2 * numpy.pi / wavelength * b * d[j + 1])
            R = (r + R * phase) / (1 + r * R * phase)
        out.append(R)
    return out


def mlayer_amplitudes(ans, anp, phase_s, phase_p):
    """MLayer amplitudes (exp(+i omega t)) converted to the SHADOW4 convention (exp(-i omega t))."""
    return ans * numpy.exp(-1j * phase_s), anp * numpy.exp(-1j * phase_p)


def get_source(energy, divergence):
    # elliptically polarized beam (P=0.7, phase difference 0.6 rad)
    s = SourceGeometrical(nrays=1000, seed=123)
    s.set_spatial_type_rectangle(width=1e-4, height=1e-4)
    s.set_angular_distribution_uniform(hdiv1=-divergence, hdiv2=divergence, vdiv1=-divergence, vdiv2=divergence)
    s.set_energy_distribution_singleline(energy)
    s.set_polarization(polarization_degree=0.7, phase_diff=0.6, coherent_beam=1)
    return s.get_beam()


def efield(beam):
    """Complex electric vector of each ray."""
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    return j0[:, None] * e_S + j1[:, None] * e_P


def get_multilayer(structure, period):
    return S4PlaneMultilayer(name="multilayer", f_refl=4, structure=structure, period=period, Gamma=0.5)


def check_multilayer(beam, oe, grazing, alpha, movements=None):
    """Returns the errors of the field and of its transversality, and the mean intensities (s, p) of the
    reflected rays relative to the incident ones."""
    element = S4PlaneMultilayerElement(optical_element=oe, movements=movements, input_beam=beam,
                                       coordinates=ElementCoordinates(p=1.0, q=1.0,
                                                                      angle_radial=numpy.pi / 2 - grazing,
                                                                      angle_radial_out=numpy.pi / 2 - grazing,
                                                                      angle_azimuthal=alpha))
    _, footprint = element.trace_beam()

    # incident beam in the (moved) multilayer frame, and reflected beam back in the moved multilayer frame
    beam_in = beam.duplicate()
    beam_in.rotate(alpha, axis=2)
    beam_in.rotate(grazing, axis=1)
    beam_in.translation([0.0, -numpy.cos(grazing), numpy.sin(grazing)]) # p=1
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
    local_grazing = numpy.arcsin(numpy.abs(vector_dot(v_in, normal)))
    pr = MLayer.initialize_from_bilayer_stack_in_compressed_format(structure=oe._structure,
                                                                   bilayer_thickness=oe._period,
                                                                   bilayer_gamma=oe._Gamma,
                                                                   use_xraylib_or_dabax=0)
    rs, rp = mlayer_amplitudes(*pr.reflectivity(numpy.degrees(local_grazing), beam_in.get_column(26)))
    s = vector_norm(vector_cross(v_in, v_out))
    p_in = vector_cross(s, v_in)
    p_out = vector_cross(s, v_out)
    E_expected = (rs * vector_dot(E_in, s))[:, None] * s + (rp * vector_dot(E_in, p_in))[:, None] * p_out

    good = beam_out.get_column(10) > 0
    scale = numpy.abs(E_in).max()
    error = numpy.abs(E_out - E_expected)[good].max() / scale
    transversality = numpy.abs(vector_dot(E_out, v_out))[good].max() / scale
    Rs = (numpy.abs(rs) ** 2).mean()
    Rp = (numpy.abs(rp) ** 2).mean()
    return error, transversality, Rs, Rp


class TestMLayerConventions(unittest.TestCase):

    def test_stack_vs_parratt(self):
        npair, t_odd, t_even = 5, 15.0, 10.0 # A, odd is the top sublayer
        do, bo, de, be, ds, bs = 2e-5, 2e-6, 5e-6, 1e-7, 8e-6, 2e-7
        wavelength = 1.24 # A
        grazing = numpy.radians(numpy.linspace(0.2, 6, 30))
        ans, anp, phase_s, phase_p = MLayer._fresnel(1.0, 1.0, npair, numpy.sin(grazing), 1.0, wavelength,
                                                     do, de, ds, bo, be, bs,
                                                     numpy.full(npair, t_odd), numpy.full(npair, t_even),
                                                     numpy.zeros(npair), numpy.zeros(npair))
        rs_ml, rp_ml = mlayer_amplitudes(ans, anp, phase_s, phase_p)
        layers = [(1 - do + 1j * bo, t_odd), (1 - de + 1j * be, t_even)] * npair
        rs, rp = parratt(grazing, wavelength, layers, 1 - ds + 1j * bs)
        assert_allclose(rs_ml, rs, atol=1e-10)
        assert_allclose(rp_ml, rp, atol=1e-10)

    def test_substrate_vs_fresnel(self):
        # no bilayers: Fresnel equations of the substrate (the odd-layer index must not be used)
        d, b = 3e-5, 4e-6
        grazing = numpy.radians(numpy.array([0.05, 0.2, 0.5, 2.0, 10.0]))
        ans, anp, phase_s, phase_p = MLayer._fresnel(1.0, 1.0, 0, numpy.sin(grazing), 1.0, 1.0,
                                                     1e-5, 0.0, d, 1e-6, 0.0, b,
                                                     numpy.array([10.0]), numpy.array([10.0]),
                                                     numpy.zeros(1), numpy.zeros(1))
        rs_ml, rp_ml = mlayer_amplitudes(ans, anp, phase_s, phase_p)
        rs, rp = PreRefl.reflectivity_amplitudes_fresnel_external(photon_energy_ev=10000.0,
                                                                  refraction_index_1=1.0 + 0j,
                                                                  refraction_index_2=1 - d + 1j * b,
                                                                  grazing_angle_mrad=grazing * 1e3, method=0)
        assert_allclose(rs_ml, rs, atol=1e-10)
        assert_allclose(rp_ml, rp, atol=1e-10)


class TestMultilayerJones(unittest.TestCase):

    def _check(self, beam, oe, grazing, alpha, movements=None):
        error, transversality, Rs, Rp = check_multilayer(beam, oe, grazing, alpha, movements=movements)
        assert_allclose(error, 0.0, atol=1e-9) # the local grazing angle is recalculated from the directions
        assert_allclose(transversality, 0.0, atol=1e-12)
        return Rs, Rp

    def test_hard_xrays(self):
        # B/W, 25 A period, 8 keV: first Bragg peak at ~1.8 deg; the divergence scans the peak
        beam = get_source(8000.0, 3e-3)
        oe = get_multilayer("[B,W]x50+Si", 25.0)
        grazing = numpy.radians(1.8)
        movements = S4BeamlineElementMovements(f_move=1, offset_x=1e-4, rotation_x=1e-4, rotation_z=0.05)
        for alpha, mov in [(0.0, None), (numpy.radians(30), None), (numpy.radians(-70), movements)]:
            Rs, Rp = self._check(beam, oe, grazing, alpha, mov)
            self.assertGreater(Rs, 0.05)

    def test_polarizer_45deg(self):
        # B/W, period lambda / (2 sin 45 deg) at 500 eV: Bragg angle ~45 deg (Brewster angle in X-rays), the
        # pi reflectivity vanishes
        energy = 500.0
        wavelength = 12398.419843320026 / energy # A
        beam = get_source(energy, 2e-3)
        oe = get_multilayer("[B,W]x30+Si", wavelength / (2 * numpy.sin(numpy.radians(45))))
        grazing = numpy.radians(45.0)
        for alpha in [0.0, numpy.radians(30)]:
            Rs, Rp = self._check(beam, oe, grazing, alpha)
            self.assertLess(Rp, 1e-3 * Rs)


if __name__ == "__main__":
    unittest.main()
