#
#
#
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical

light_source = SourceGeometrical(name='SourceGeometrical', nrays=15000, seed=5676561)
light_source.set_spatial_type_ellipse(width=0.247329, height=0.247329)
light_source.set_angular_distribution_flat(hdiv1=-0.000000, hdiv2=0.000000, vdiv1=-0.000000, vdiv2=0.000000)
light_source.set_energy_distribution_singleline(1000.000000, unit='eV')
light_source.set_polarization(polarization_degree=1.000000, phase_diff=0.000000, coherent_beam=1)
beam = light_source.get_beam()

# optical element number XX
from syned.beamline.shape import Ellipse

boundary_shape = Ellipse(a_axis_min=-0.12174, a_axis_max=0.12174, b_axis_min=-0.12174, b_axis_max=0.12174)

from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen

optical_element = S4Screen(name='slit1', boundary_shape=boundary_shape,
                           i_abs=0, i_stop=1, thick=0, file_abs='')

from syned.beamline.element_coordinates import ElementCoordinates

coordinates = ElementCoordinates(p=0, q=0, angle_radial=0, angle_azimuthal=0, angle_radial_out=0)
from shadow4.beamline.optical_elements.absorbers.s4_screen import S4ScreenElement

beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)

beam, footprint = beamline_element.trace_beam()

# optical element number XX
boundary_shape = None

from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror

optical_element = S4ConicMirror(name='Conic coefficients Mirror', boundary_shape=boundary_shape,
                                conic_coefficients=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00389288,
                                                    -3.7886286736e-06],
                                f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j,
                                coating_material='Si', coating_density=2.33, coating_roughness=0)

from syned.beamline.element_coordinates import ElementCoordinates

coordinates = ElementCoordinates(p=10000, q=-3.80610854019, angle_radial=-5.1034731996e-12,
                                 angle_azimuthal=3.14159265359, angle_radial_out=-5.1034731996e-12)
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirrorElement

beamline_element = S4ConicMirrorElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)

beam, mirr = beamline_element.trace_beam()

# optical element number XX
boundary_shape = None

from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirror

optical_element = S4ConicMirror(name='Conic coefficients Mirror', boundary_shape=boundary_shape,
                                conic_coefficients=[-539.3838807775953, -539.3838807775953, 1.1044872307738431, 0.0,
                                                    0.0, 0.0, 0.0, 0.0, -2.104042652188017, 0.002047683051228155],
                                f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j,
                                coating_material='Si', coating_density=2.33, coating_roughness=0)

from syned.beamline.element_coordinates import ElementCoordinates

coordinates = ElementCoordinates(p=3.80610854019, q=1.904995, angle_radial=-5.1034731996e-12, angle_azimuthal=0,
                                 angle_radial_out=-5.1034731996e-12)
from shadow4.beamline.optical_elements.mirrors.s4_conic_mirror import S4ConicMirrorElement

beamline_element = S4ConicMirrorElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)

beam, mirr = beamline_element.trace_beam()

# test plot
if True:
    from srxraylib.plot.gol import plot_scatter

    plot_scatter(beam.get_photon_energy_eV(nolost=1), beam.get_column(23, nolost=1), title='(Intensity,Photon Energy)',
                 plot_histograms=0)
    plot_scatter(1e6 * beam.get_column(1, nolost=1), 1e6 * beam.get_column(3, nolost=1), title='(X,Z) in microns')