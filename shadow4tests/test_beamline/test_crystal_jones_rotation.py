__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Regression test for the Jones rotation R(alpha) applied in crystal diffraction.
# R must be the passive (projection) matrix R_jk = (scattering-plane versor j) . (incident versor k);
# using its transpose rotates the field the wrong way for alpha not multiple of 90 deg.
#

import unittest
import numpy
from numpy.testing import assert_allclose

from shadow4.beam.s4_beam import S4Beam
from shadow4.tools.arrayofvectors import vector_cross, vector_norm, vector_dot
from shadow4.beamline.optical_elements.crystals.s4_crystal import S4CrystalElement
from shadow4.beamline.optical_elements.mosaic_crystals.s4_mosaic_crystal import S4MosaicCrystalElement

METHODS = {
    "S4CrystalElement": S4CrystalElement._calculate_jones_and_efield_directions,
    "S4MosaicCrystalElement": S4MosaicCrystalElement._calculate_jones_and_efield_directions,
}

THETA = numpy.radians(20.0)
V_IN = numpy.array([[0.0, numpy.cos(THETA), -numpy.sin(THETA)]])
V_OUT = numpy.array([[0.0, numpy.cos(THETA), numpy.sin(THETA)]])
SIGMA_I = vector_norm(vector_cross(V_IN, V_OUT))   # perpendicular to the scattering plane
PI_I = vector_norm(vector_cross(SIGMA_I, V_IN))


def make_beam(alpha, j0, j1):
    # incident sigma versor rotated by alpha from SIGMA_I toward PI_I; pi = sigma x v (SHADOW convention)
    e_S = numpy.cos(alpha) * SIGMA_I + numpy.sin(alpha) * PI_I
    e_P = vector_norm(vector_cross(e_S, V_IN))
    beam = S4Beam.initialize_as_pencil(N=1)
    beam.set_column(4, V_IN[:, 0])
    beam.set_column(5, V_IN[:, 1])
    beam.set_column(6, V_IN[:, 2])
    beam.set_jones_components(numpy.array([j0]), numpy.array([j1]), e_S=e_S, e_P=e_P)
    return beam


def field_out(method, beam, r_s, r_p):
    j0, j1, ee_S, ee_P = method(None, beam, None, V_IN, V_OUT, numpy.array([r_s]), numpy.array([r_p]))
    return j0[:, None] * ee_S + j1[:, None] * ee_P


def field_expected(beam, r_s, r_p):
    j0, j1 = beam.get_jones_components()
    e_S, e_P = beam.get_efield_directions()
    E_in = j0[:, None] * e_S + j1[:, None] * e_P
    ee_P = vector_norm(vector_cross(SIGMA_I, V_OUT))
    return r_s * vector_dot(E_in, SIGMA_I)[:, None] * SIGMA_I + r_p * vector_dot(E_in, PI_I)[:, None] * ee_P


class TestCrystalJonesRotation(unittest.TestCase):

    def test_field_matches_projection(self):
        rng = numpy.random.default_rng(1)
        r_s = 0.9 * numpy.exp(1j * 0.3)
        r_p = 0.4 * numpy.exp(-1j * 1.1)
        for name, method in METHODS.items():
            for alpha_deg in [0, 15, 30, 45, 60, 90, 135, 180, 250, 315]:
                j0 = abs(rng.normal()) + 0.1
                j1 = (abs(rng.normal()) + 0.1) * numpy.exp(1j * rng.uniform(0, 2 * numpy.pi))
                beam = make_beam(numpy.radians(alpha_deg), j0, j1)
                assert_allclose(field_out(method, beam, r_s, r_p), field_expected(beam, r_s, r_p),
                                atol=1e-12, err_msg="%s alpha=%d deg" % (name, alpha_deg))

    def test_polariser_at_45deg(self):
        # beam sigma at 45 deg to the crystal sigma; field linearly polarised along the crystal sigma
        # (passes completely when r_pi=0) or along the crystal pi (fully extinguished).
        alpha = numpy.radians(45.0)
        c, s = numpy.cos(alpha), numpy.sin(alpha)
        for name, method in METHODS.items():
            beam = make_beam(alpha, c, -s)   # E = c e_S - s e_P = SIGMA_I
            I = numpy.sum(numpy.abs(field_out(method, beam, 1.0, 0.0)) ** 2)
            self.assertAlmostEqual(I, 1.0, places=12, msg=name)

            beam = make_beam(alpha, s, c)    # E = s e_S + c e_P = PI_I
            I = numpy.sum(numpy.abs(field_out(method, beam, 1.0, 0.0)) ** 2)
            self.assertAlmostEqual(I, 0.0, places=12, msg=name)


class TestCrystalJonesForwardScattering(unittest.TestCase):

    def test_forward_scattering(self):
        # vOut == vIn: scattering plane undefined -> incident sigma/pi kept (R = identity), no NaN.
        # Rays 0,2 are forward-scattered, ray 1 is a normal Bragg reflection.
        r_s = numpy.array([0.9 * numpy.exp(1j * 0.3)] * 3)
        r_p = numpy.array([0.4 * numpy.exp(-1j * 1.1)] * 3)
        alpha = numpy.radians(30.0)
        e_S0 = numpy.cos(alpha) * SIGMA_I + numpy.sin(alpha) * PI_I
        e_S = numpy.repeat(e_S0, 3, axis=0)
        v_in = numpy.repeat(V_IN, 3, axis=0)
        v_out = v_in.copy()
        v_out[1] = V_OUT[0]
        e_P = vector_norm(vector_cross(e_S, v_in))
        j0 = numpy.array([0.6 + 0.1j, 0.5, 0.3j])
        j1 = numpy.array([0.2 - 0.7j, 0.4j, 0.8])

        beam = S4Beam.initialize_as_pencil(N=3)
        for k in range(3):
            beam.set_column(4 + k, v_in[:, k])
        beam.set_jones_components(j0, j1, e_S=e_S, e_P=e_P)

        for name, method in METHODS.items():
            o0, o1, ee_S, ee_P = method(None, beam, None, v_in, v_out, r_s, r_p)
            self.assertFalse(numpy.isnan(o0).any() or numpy.isnan(o1).any(), msg=name)
            for k in [0, 2]:
                assert_allclose(ee_S[k], e_S[k], atol=1e-12, err_msg=name)
                assert_allclose(ee_P[k], e_P[k], atol=1e-12, err_msg=name)
                assert_allclose([o0[k], o1[k]], [r_s[k] * j0[k], r_p[k] * j1[k]], atol=1e-12, err_msg=name)
            assert_allclose(ee_S[1], SIGMA_I[0], atol=1e-12, err_msg=name)


class TestJonesConventions(unittest.TestCase):
    # SHADOW convention: Jones components along (u_s, u_p), u_p = u_s x v (left-handed).
    # Standard convention: along (u_s, -u_p) (right-handed). Jones matrices: J_shadow = F J_std F, F=diag(1,-1).

    def test_get_set_use_standard(self):
        j0, j1 = 0.6 * numpy.exp(0.4j), 0.8 * numpy.exp(-1.2j)
        beam = make_beam(numpy.radians(20.0), j0, j1)            # set in SHADOW convention
        s0, s1 = beam.get_jones_components(use_standard=1)
        assert_allclose([s0[0], s1[0]], [j0, -j1], atol=1e-12)
        beam.set_jones_components(s0, s1, use_standard=1)          # round trip
        assert_allclose(beam.get_jones()[0], [j0, j1], atol=1e-12)
        assert_allclose(beam.get_jones(use_standard=1)[0], [j0, -j1], atol=1e-12)

    def test_off_diagonal_jones_matrix(self):
        rng = numpy.random.default_rng(5)
        ee_P = vector_norm(vector_cross(SIGMA_I, V_OUT))
        for name, method in METHODS.items():
            for alpha_deg in [0, 30, 45, 90, 200]:
                J = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                beam = make_beam(numpy.radians(alpha_deg), rng.normal() + 1j * rng.normal(),
                                 rng.normal() + 1j * rng.normal())
                j0, j1 = beam.get_jones_components()
                e_S, e_P = beam.get_efield_directions()
                E_in = j0[:, None] * e_S + j1[:, None] * e_P

                for use_standard, sign in [(1, -1.0), (0, 1.0)]:
                    # expected: project E_in on (SIGMA_I, sign*PI_I), apply J, rebuild on (SIGMA_I, sign*ee_P)
                    a = numpy.array([vector_dot(E_in, SIGMA_I)[0], sign * vector_dot(E_in, PI_I)[0]])
                    b = J @ a
                    expected = b[0] * SIGMA_I + b[1] * sign * ee_P
                    o0, o1, s_S, s_P = method(None, beam, None, V_IN, V_OUT,
                                              numpy.array([J[0, 0]]), numpy.array([J[1, 1]]),
                                              r_SP=numpy.array([J[0, 1]]), r_PS=numpy.array([J[1, 0]]),
                                              use_standard=use_standard)
                    got = o0[:, None] * s_S + o1[:, None] * s_P
                    assert_allclose(got, expected, atol=1e-12,
                                    err_msg="%s alpha=%d use_standard=%d" % (name, alpha_deg, use_standard))


class TestCrystalStandardJones(unittest.TestCase):
    # method_efields_management=2: standard Jones algebra (angle chi, passive R(chi), right-handed frames).
    # It must give the same electric field as the default calculation in the SHADOW frame.

    def test_standard_vs_shadow(self):
        rng = numpy.random.default_rng(11)
        for alpha_deg in [0, 30, 45, 90, 135, 250]:
            J = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))   # standard convention
            beam = make_beam(numpy.radians(alpha_deg), rng.normal() + 1j * rng.normal(),
                             rng.normal() + 1j * rng.normal())
            args = (V_IN, V_OUT, numpy.array([J[0, 0]]), numpy.array([J[1, 1]]))
            kw = dict(r_SP=numpy.array([J[0, 1]]), r_PS=numpy.array([J[1, 0]]))
            s0, s1, s_S, s_P = S4CrystalElement._calculate_jones_and_efield_directions_standard(None, beam, *args, **kw)
            o0, o1, o_S, o_P = S4CrystalElement._calculate_jones_and_efield_directions(None, beam, None, *args,
                                                                                     use_standard=1, **kw)
            field_standard = s0[:, None] * s_S + s1[:, None] * (-s_P)       # components along (u_s,f, -u_p,f)
            field_shadow = o0[:, None] * o_S + o1[:, None] * o_P
            assert_allclose(field_standard, field_shadow, atol=1e-12, err_msg="alpha=%d" % alpha_deg)

    def test_trace_method_0_vs_2(self):
        # full trace: Si 111 at 8 keV, crystal rotated 45 deg around the beam, elliptically polarized source
        from syned.beamline.element_coordinates import ElementCoordinates
        from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
        from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal, S4PlaneCrystalElement

        src = SourceGeometrical(nrays=200, seed=1234)
        src.set_energy_distribution_singleline(value=8000, unit='eV')
        src.set_spatial_type_rectangle(width=1e-3, height=1e-3)
        src.set_angular_distribution_uniform(-20e-6, 20e-6, -20e-6, 20e-6)
        src.set_polarization(polarization_degree=0.7, phase_diff=numpy.pi / 3, coherent_beam=True)
        beam = src.get_beam()

        out = {}
        for method in [0, 2]:
            oe = S4PlaneCrystal(name="xtal", material="Si", miller_index_h=1, miller_index_k=1, miller_index_l=1,
                                thickness=0.010, f_central=True, f_phot_cent=0, phot_cent=8000.0,
                                material_constants_library_flag=1, method_efields_management=method)
            el = S4PlaneCrystalElement(optical_element=oe,
                                       coordinates=ElementCoordinates(p=1.0, q=1.0, angle_radial=0.0,
                                                                      angle_azimuthal=numpy.radians(45.0),
                                                                      angle_radial_out=0.0),
                                       input_beam=beam)
            b, _ = el.trace_beam()
            j0, j1 = b.get_jones_components()
            e_S, e_P = b.get_efield_directions()
            out[method] = (j0[:, None] * e_S + j1[:, None] * e_P, b.get_columns([30, 31, 32, 33]))

        self.assertGreater(numpy.abs(out[0][1][0]).max(), 0.1)          # something is reflected
        assert_allclose(out[2][0], out[0][0], atol=1e-10)                # same complex electric field
        assert_allclose(out[2][1], out[0][1], atol=1e-10)                # same Stokes parameters


if __name__ == "__main__":
    unittest.main()
