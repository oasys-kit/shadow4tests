__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Mosaic crystals: the S/P polarization channel must be chosen with the intensities along the sigma and pi
# directions of the scattering plane, not of the beam. Before the fix, the Monte Carlo model gave zero
# reflectivity for a sigma-polarized beam on a crystal rotated by 90 deg around the beam (field along pi).
#
# Handedness and coherence (TestMosaicHandedness): the sigma and pi waves are scattered coherently, with the
# polarization factor C = cos(2 theta_B) for pi. A circularly polarized beam keeps its handedness for theta_B < 45 deg
# and reverses it for theta_B > 45 deg (C < 0), as for perfect crystals. Before the fix, the macroscopic model lost
# the sign of C (no reversal), and the Monte Carlo model traced each ray in a single polarization channel, so the
# reflected rays were linearly polarized (S3 = 0).
#

import unittest
import numpy
from numpy.testing import assert_allclose

from shadow4.beam.s4_beam import S4Beam
from shadow4.tools.arrayofvectors import vector_cross, vector_norm
from shadow4.beamline.optical_elements.mosaic_crystals.s4_mosaic_crystal import S4MosaicCrystalElement


class TestSigmaIntensityInScatteringPlane(unittest.TestCase):

    def test_projection(self):
        theta = numpy.radians(20.0)
        v_in = numpy.array([[0.0, numpy.cos(theta), -numpy.sin(theta)]])
        normal = numpy.array([[0.0, 0.0, 1.0]])
        x = numpy.array([[1.0, 0.0, 0.0]])                          # sigma of the scattering plane (y,z)
        pi_x = vector_norm(vector_cross(x, v_in))
        a_s, a_p = 0.6, 0.8 * numpy.exp(0.7j)
        for alpha_deg in [0, 30, 45, 90, 180, 270]:
            alpha = numpy.radians(alpha_deg)
            e_S = numpy.cos(alpha) * x + numpy.sin(alpha) * pi_x   # beam sigma rotated by alpha
            e_P = vector_norm(vector_cross(e_S, v_in))
            beam = S4Beam.initialize_as_pencil(N=1)
            for k in range(3):
                beam.set_column(4 + k, v_in[:, k])
            beam.set_jones_components(numpy.array([a_s]), numpy.array([a_p]), e_S=e_S, e_P=e_P)
            Is = S4MosaicCrystalElement._sigma_intensity_in_scattering_plane(None, beam, normal)
            expected = abs(a_s * numpy.cos(alpha) - a_p * numpy.sin(alpha)) ** 2   # E . x
            assert_allclose(Is, [expected], atol=1e-12, err_msg="alpha=%d" % alpha_deg)


class TestMosaicMonteCarloRotatedCrystal(unittest.TestCase):

    def test_rotated_crystal(self):
        # HOPG 002 at 8 keV, sigma-polarized source (P=1). At alpha=90 deg the field is along the crystal pi.
        from dabax.dabax_xraylib import DabaxXraylib
        from syned.beamline.element_coordinates import ElementCoordinates
        from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
        from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import \
            S4PlaneMosaicCrystal, S4PlaneMosaicCrystalElement

        def run(alpha_deg, method, nrays=2000):
            numpy.random.seed(1)
            src = SourceGeometrical(nrays=nrays, seed=5676561)
            src.set_spatial_type_point()
            src.set_depth_distribution_off()
            src.set_angular_distribution_uniform(hdiv1=-1e-5, hdiv2=1e-5, vdiv1=-1e-5, vdiv2=1e-5)
            src.set_energy_distribution_singleline(8000.0, unit='eV')
            src.set_polarization(polarization_degree=1.0, phase_diff=0, coherent_beam=0)
            oe = S4PlaneMosaicCrystal(name='m', material='Graphite',
                                      miller_index_h=0, miller_index_k=0, miller_index_l=2,
                                      thickness=0.001, f_central=1, f_phot_cent=0, phot_cent=8000.0,
                                      material_constants_library_flag=1,
                                      dabax=DabaxXraylib(file_f0="f0_InterTables.dat", file_f1f2="f1f2_Windt.dat"),
                                      mosaicity_fwhm_deg=0.2, mosaicity_profile_flag=0, calculation_method=method)
            coordinates = ElementCoordinates(p=10, q=10, angle_radial=1.337226398,
                                             angle_azimuthal=numpy.radians(alpha_deg),
                                             angle_radial_out=1.337226398)
            out, _ = S4PlaneMosaicCrystalElement(optical_element=oe, coordinates=coordinates,
                                                 input_beam=src.get_beam()).trace_beam()
            return out.get_column(23, nolost=1).sum() / nrays

        analytic_90 = run(90, 0)
        mc_0 = run(0, 1)
        mc_90 = run(90, 1)
        self.assertGreater(analytic_90, 0.1)
        self.assertAlmostEqual(mc_90 / analytic_90, 1.0, delta=0.1)
        # pi reflectivity is lower than sigma (cos(2 theta_B) < 1)
        self.assertLess(mc_90, mc_0 * 1.05)


def trace_hopg(l, method, polarization_degree, phase_diff, nrays=3000):
    """HOPG 00l at 8 keV, aligned; returns the incident and reflected beams."""
    from dabax.dabax_xraylib import DabaxXraylib
    from syned.beamline.element_coordinates import ElementCoordinates
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    from shadow4.beamline.optical_elements.mosaic_crystals.s4_plane_mosaic_crystal import \
        S4PlaneMosaicCrystal, S4PlaneMosaicCrystalElement
    numpy.random.seed(1)
    src = SourceGeometrical(nrays=nrays, seed=5676561)
    src.set_spatial_type_point()
    src.set_depth_distribution_off()
    src.set_angular_distribution_uniform(hdiv1=-1e-5, hdiv2=1e-5, vdiv1=-1e-5, vdiv2=1e-5)
    src.set_energy_distribution_singleline(8000.0, unit='eV')
    src.set_polarization(polarization_degree=polarization_degree, phase_diff=phase_diff, coherent_beam=1)
    beam = src.get_beam()
    oe = S4PlaneMosaicCrystal(name='m', material='Graphite', miller_index_h=0, miller_index_k=0, miller_index_l=l,
                              thickness=0.001, f_central=1, f_phot_cent=0, phot_cent=8000.0,
                              material_constants_library_flag=1,
                              dabax=DabaxXraylib(file_f0="f0_InterTables.dat", file_f1f2="f1f2_Windt.dat"),
                              mosaicity_fwhm_deg=0.2, mosaicity_profile_flag=0, calculation_method=method)
    element = S4PlaneMosaicCrystalElement(optical_element=oe, input_beam=beam,
                                          coordinates=ElementCoordinates(p=10, q=10, angle_radial=0.0,
                                                                         angle_radial_out=0.0))
    element.set_crystalpy_diffraction_setup()
    element.align_crystal()
    theta_B = numpy.pi / 2 - element.get_coordinates().angle_radial()
    out, _ = element.trace_beam()
    return beam, out, theta_B


def beam_stokes(beam):
    """Stokes parameters (S0, S1, S2, S3) of the good rays, normalized to S0."""
    good = beam.get_column(10) > 0
    S = numpy.array([beam.get_column(c)[good].sum() for c in (30, 31, 32, 33)])
    return S / S[0]


class TestMosaicHandedness(unittest.TestCase):

    def test_circular_polarization(self):
        # incident beam circularly polarized (S3 = -1); HOPG 002 (theta_B = 13.4 deg) and 008 (theta_B = 67.8 deg)
        for l in [2, 8]:
            stokes = {}
            for method in [0, 1]: # 0: macroscopic, 1: Monte Carlo
                beam, out, theta_B = trace_hopg(l, method, 0.5, numpy.pi / 2)
                assert_allclose(beam_stokes(beam)[3], -1.0, atol=1e-12)
                stokes[method] = beam_stokes(out)
                msg = "HOPG 00%d method=%d" % (l, method)
                # the handedness is kept (C > 0) or reversed (C < 0), and the beam stays polarized
                self.assertEqual(numpy.sign(stokes[method][3]), -numpy.sign(numpy.cos(2 * theta_B)), msg=msg)
                self.assertGreater(numpy.abs(stokes[method][3]), 0.9, msg=msg)
                self.assertGreater(numpy.sum(stokes[method][1:] ** 2), 0.999, msg=msg) # fully polarized
            # Monte Carlo: same r_pi / r_sigma as the macroscopic model (approximation)
            assert_allclose(stokes[1], stokes[0], atol=1e-3, err_msg="HOPG 00%d" % l)

    def test_monte_carlo_intensity_unbiased(self):
        # the reflected intensity of a circularly polarized beam is the mean of those of the sigma- and
        # pi-polarized beams (each ray carries both amplitudes, normalized to the intensity of the traced channel)
        for l in [2, 8]:
            I = {}
            for P, phase in [(1.0, 0.0), (0.0, 0.0), (0.5, numpy.pi / 2)]:
                beam, out, _ = trace_hopg(l, 1, P, phase, nrays=5000)
                I[P] = out.get_column(23, nolost=1).sum() / beam.get_column(23).sum()
            self.assertGreater(I[1.0], I[0.0])
            self.assertAlmostEqual(I[0.5] / (0.5 * (I[1.0] + I[0.0])), 1.0, delta=0.05, msg="HOPG 00%d" % l)


if __name__ == "__main__":
    unittest.main()
