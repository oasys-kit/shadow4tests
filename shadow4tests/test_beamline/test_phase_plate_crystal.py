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


if __name__ == "__main__":
    unittest.main()
