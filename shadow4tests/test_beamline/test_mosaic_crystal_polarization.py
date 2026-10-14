__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Mosaic crystals: the S/P polarization channel must be chosen with the intensities along the sigma and pi
# directions of the scattering plane, not of the beam. Before the fix, the Monte Carlo model gave zero
# reflectivity for a sigma-polarized beam on a crystal rotated by 90 deg around the beam (field along pi).
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


if __name__ == "__main__":
    unittest.main()
