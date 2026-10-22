__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Phase plate crystal (S4PlanePhasePlateCrystalElement): transmitted (forward-diffracted) beam of a perfect crystal
# plate in Bragg geometry, detuned from the Bragg angle (x-ray phase retarder).
#
# The sigma/pi amplitudes (t_s, t_p) are compared with crystalpy (BraggTransmission), and the output Stokes
# parameters of a plate rotated around the beam by alpha=+-45 deg are compared with the Jones calculus:
# for a horizontally (sigma) polarized input, the output Jones vector in the SHADOW basis is
# (t_s cos^2(a) + t_p sin^2(a), (t_s - t_p) sin(a) cos(a)), which gives (in the frame of the plate) the
# well-known circular polarization P3 = +-2|t_s||t_p| sin(phi_p - phi_s) / (|t_s|^2 + |t_p|^2).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from syned.beamline.element_coordinates import ElementCoordinates
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.phase_plate_crystals.s4_plane_phase_plate_crystal import \
    S4PlanePhasePlateCrystal, S4PlanePhasePlateCrystalElement

from crystalpy.diffraction.DiffractionSetupXraylib import DiffractionSetupXraylib
from crystalpy.diffraction.GeometryType import BraggTransmission
from crystalpy.diffraction.Diffraction import Diffraction
from crystalpy.util.ComplexAmplitudePhoton import ComplexAmplitudePhoton

ENERGY = 10000.0
THICKNESS = 100e-6
HKL = (2, 2, 0)


def get_source(P, nrays=5):
    s = SourceGeometrical(nrays=nrays, seed=1)
    s.set_spatial_type_point()
    s.set_angular_distribution_flat(0, 0, 0, 0)
    s.set_energy_distribution_singleline(ENERGY)
    s.set_polarization(polarization_degree=P, phase_diff=0.0, coherent_beam=1)
    return s.get_beam()


def trace_plate(beam, alpha=0.0, f_detune=2, detune_angle=0.0, detune_half_darwin_widths=6.0):
    oe = S4PlanePhasePlateCrystal(material="Si", miller_index_h=HKL[0], miller_index_k=HKL[1], miller_index_l=HKL[2],
                                  thickness=THICKNESS, f_central=1, phot_cent=ENERGY,
                                  f_detune=f_detune, detune_angle=detune_angle, detune_half_darwin_widths=detune_half_darwin_widths)
    el = S4PlanePhasePlateCrystalElement(optical_element=oe,
                                         coordinates=ElementCoordinates(p=1, q=1, angle_azimuthal=alpha),
                                         input_beam=beam)
    out, footprint = el.trace_beam()
    return out, el


def crystalpy_amplitudes(detune_half_darwin_widths):
    setup = DiffractionSetupXraylib(geometry_type=BraggTransmission(), crystal_name="Si", thickness=THICKNESS,
                                    miller_h=HKL[0], miller_k=HKL[1], miller_l=HKL[2],
                                    asymmetry_angle=0.0, azimuthal_angle=0.0)
    darwin_half_width = setup.darwinHalfwidthS(ENERGY)
    v = setup.vectorIncomingPhotonDirection(ENERGY, detune_half_darwin_widths * darwin_half_width, angle_center_flag=1)
    photon = ComplexAmplitudePhoton(energy_in_ev=ENERGY, direction_vector=v, Esigma=1.0, Epi=1.0)
    c = Diffraction.calculateDiffractedComplexAmplitudes(setup, photon, calculation_method=1, is_thick=0)
    return complex(numpy.ravel(c["S"])[0]), complex(numpy.ravel(c["P"])[0]), darwin_half_width


class TestPhasePlateCrystal(unittest.TestCase):

    def test_amplitudes_vs_crystalpy(self):
        for nhdw in [-6.0, -5.0, 6.0, 10.0]:
            t_s, t_p, _ = crystalpy_amplitudes(nhdw)
            out_s, _ = trace_plate(get_source(1.0), detune_half_darwin_widths=nhdw)
            out_p, _ = trace_plate(get_source(0.0), detune_half_darwin_widths=nhdw)
            assert_allclose(out_s.get_column(23), abs(t_s) ** 2, rtol=1e-6)
            assert_allclose(out_p.get_column(23), abs(t_p) ** 2, rtol=1e-6)

    def test_undeviated_beam(self):
        beam = get_source(1.0)
        out, el = trace_plate(beam)
        coor = el.get_coordinates()
        assert_allclose(coor.angle_radial_out(), numpy.pi - coor.angle_radial(), atol=1e-12)
        assert_allclose(out.get_columns([4, 5, 6]), beam.get_columns([4, 5, 6]), atol=1e-12)
        # electric vectors remain perpendicular to the ray
        v = out.get_columns([4, 5, 6])
        for cols in ([7, 8, 9], [16, 17, 18]):
            assert_allclose(numpy.sum(v * out.get_columns(cols), axis=0), 0.0, atol=1e-12)

    def test_circular_polarization(self):
        t_s, t_p, _ = crystalpy_amplitudes(6.0)
        s0 = 0.5 * (abs(t_s) ** 2 + abs(t_p) ** 2)
        p1 = (abs(t_s) ** 2 - abs(t_p) ** 2) / (abs(t_s) ** 2 + abs(t_p) ** 2)
        p2 = 2 * abs(t_s) * abs(t_p) * numpy.cos(numpy.angle(t_p / t_s)) / (abs(t_s) ** 2 + abs(t_p) ** 2)
        p3 = 2 * abs(t_s) * abs(t_p) * numpy.sin(numpy.angle(t_p / t_s)) / (abs(t_s) ** 2 + abs(t_p) ** 2)
        self.assertGreater(abs(p3), 0.95)  # nearly a quarter-wave plate at 6 Darwin half widths
        for sign in [1, -1]:
            out, _ = trace_plate(get_source(1.0), alpha=sign * numpy.radians(45))
            assert_allclose(out.get_column(30), s0, rtol=1e-6)
            assert_allclose(out.get_column(42), p1, atol=1e-6)
            assert_allclose(numpy.abs(out.get_column(43)), abs(p2), atol=1e-6)
            assert_allclose(numpy.abs(out.get_column(44)), abs(p3), atol=1e-6)
        # opposite helicities for alpha = +45 and -45 deg
        out_plus, _ = trace_plate(get_source(1.0), alpha=numpy.radians(45))
        out_minus, _ = trace_plate(get_source(1.0), alpha=numpy.radians(-45))
        assert_allclose(out_plus.get_column(44), -out_minus.get_column(44), atol=1e-10)

    def test_detuning_options(self):
        _, _, darwin_half_width = crystalpy_amplitudes(0.0)
        _, el = trace_plate(get_source(1.0), f_detune=0)
        self.assertEqual(el.get_detuning_angle(ENERGY), 0.0)
        _, el = trace_plate(get_source(1.0), f_detune=1, detune_angle=50e-6)
        assert_allclose(el.get_detuning_angle(ENERGY), 50e-6)
        _, el = trace_plate(get_source(1.0), f_detune=2, detune_half_darwin_widths=-5.0)
        assert_allclose(el.get_detuning_angle(ENERGY), -5.0 * darwin_half_width, rtol=1e-6)
        # f_detune=1 with the same angle as f_detune=2 gives the same transmission
        out1, _ = trace_plate(get_source(1.0), f_detune=1, detune_angle=-5.0 * darwin_half_width)
        out2, _ = trace_plate(get_source(1.0), f_detune=2, detune_half_darwin_widths=-5.0)
        assert_allclose(out1.get_column(23), out2.get_column(23), rtol=1e-6)


class TestPhasePlateRetardationTheory(unittest.TestCase):
    #
    # Independent check (not using the crystalpy amplitudes) of the sign and value of the phase retardation
    # phi_s - phi_p = arg(t_s) - arg(t_p) of the transmitted beam, against the two-beam dynamical theory far from the
    # Bragg angle (time factor exp(-i omega t), as in SHADOW4; fields exp(i k.r)).
    #
    # The fundamental equations of the two-beam theory give, for the forward wave of wavevector k0 inside the crystal
    # (vacuum wavenumber K, polarization factor C = 1 for sigma and C = cos(2 theta_B) for pi):
    #
    #     (k0^2 - K^2) / K^2 = chi_0 + C^2 chi_h chi_hbar / (alpha - chi_0)
    #
    # where alpha = (h^2 + 2 K0.h) / K^2 is the deviation from the Bragg condition. For the symmetric Bragg case
    # (|h| = 2 K sin theta_B, K0.h = -K |h| sin theta), alpha = 4 sin theta_B (sin theta_B - sin theta)
    # = -2 sin(2 theta_B) Delta_theta, with Delta_theta = theta - theta_B; with the refraction correction,
    # alpha - chi_0 = -2 sin(2 theta_B) Delta_theta, with Delta_theta measured from the corrected Bragg angle
    # (the detuning of the plate). Far from the Bragg angle (|Delta_theta| >> Darwin width), the forward wave
    # accumulates along the thickness T (path T / gamma_0, gamma_0 = sin theta_B) the phase
    #
    #     phi = K T / (2 gamma_0) [chi_0 + C^2 chi_h chi_hbar / (alpha - chi_0)]
    #
    # and the retardation between sigma and pi is
    #
    #     phi_s - phi_p = K T (1 - cos^2(2 theta_B)) Re(chi_h chi_hbar) / (2 gamma_0 (alpha - chi_0))
    #
    # For a centrosymmetric crystal chi_h chi_hbar > 0 (the conjugation between the exp(-i omega t) and exp(+i omega t)
    # conventions does not change Re(chi_h chi_hbar)), so phi_s - phi_p > 0 on the low-angle side (Delta_theta < 0)
    # and < 0 on the high-angle side, decreasing as 1/Delta_theta. Physically: below the Bragg "band gap", the
    # coupling increases the effective refraction index of the forward wave, more for sigma (C = 1) than for pi.
    #
    # For an incident beam linearly polarized at 45 deg (SHADOW Jones vector (1, 1)/sqrt(2)) and the plate at
    # alpha = 0, the transmitted Jones vector is (t_s, t_p)/sqrt(2) and S3/S0 = sin(phi_s - phi_p) |t_s||t_p| * 2 /
    # (|t_s|^2 + |t_p|^2) (S3 > 0 is LCP, see the paper): the low-angle side gives S3 > 0.
    #

    def test_retardation_far_from_bragg(self):
        import scipy.constants as codata
        beam = get_source(0.5) # linear polarization at 45 deg: equal sigma and pi, in phase
        for nhdw in [-40.0, -20.0, -10.0, 10.0, 20.0, 40.0]:
            out, el = trace_plate(beam, detune_half_darwin_widths=nhdw)
            j0, j1 = out.get_jones_components()          # (t_s, t_p) / sqrt(2) (alpha = 0)
            retardation = numpy.angle(j0[0] / j1[0])     # phi_s - phi_p from SHADOW4 (crystalpy)

            # two-beam theory far from the Bragg angle (see the comments above)
            setup = el._crystalpy_diffraction_setup
            theta_B = float(numpy.ravel(setup.angleBraggCorrected(ENERGY))[0])
            delta_theta = el.get_detuning_angle(ENERGY)
            chi_h_chi_hbar = complex(numpy.ravel(setup.psiH(ENERGY))[0]) * \
                             complex(numpy.ravel(setup.psiH_bar(ENERGY))[0])
            K = 2 * numpy.pi / (codata.h * codata.c / codata.e / ENERGY)
            gamma_0 = numpy.sin(theta_B)
            alpha_minus_chi_0 = -2 * numpy.sin(2 * theta_B) * delta_theta
            expected = K * THICKNESS * (1 - numpy.cos(2 * theta_B) ** 2) * numpy.real(chi_h_chi_hbar) / \
                       (2 * gamma_0 * alpha_minus_chi_0)

            msg = "detuning %g half Darwin widths" % nhdw
            self.assertEqual(numpy.sign(retardation), -numpy.sign(nhdw), msg=msg)  # > 0 on the low-angle side
            assert_allclose(retardation, expected, rtol=0.01, err_msg=msg)
            # handedness: S3/S0 = sin(phi_s - phi_p) * 2 |t_s||t_p| / (|t_s|^2 + |t_p|^2)
            P3 = out.get_column(44)[0]
            assert_allclose(P3, numpy.sin(retardation) * 2 * abs(j0[0] * j1[0]) / (abs(j0[0]) ** 2 + abs(j1[0]) ** 2),
                            atol=1e-10, err_msg=msg)


if __name__ == "__main__":
    unittest.main()
