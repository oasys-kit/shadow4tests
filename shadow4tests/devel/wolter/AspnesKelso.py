from srxraylib.plot.gol import plot_scatter, set_qt
from shadow4tests.compatibility.beam3 import Beam3
from shadow4.sources.source_geometrical.source_grid_polar import SourceGridPolar
import numpy

def source_cone(do_plot=0):
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='SourceGeometrical', nrays=5000, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_angular_distribution_collimated()
    light_source.set_energy_distribution_singleline(1000.000000, unit='eV')
    light_source.set_polarization(polarization_degree=1.000000, phase_diff=0.000000, coherent_beam=0)
    beam = light_source.get_beam()
    return beam

def source_circle(do_plot=0):
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='SourceGeometrical', nrays=5000, seed=5676561)
    light_source.set_spatial_type_ellipse(width=0.002000, height=0.002000)
    light_source.set_angular_distribution_flat(hdiv1=0.000000, hdiv2=0.000000, vdiv1=0.000000, vdiv2=0.000000)
    light_source.set_energy_distribution_singleline(1000.000000, unit='eV')
    light_source.set_polarization(polarization_degree=1.000000, phase_diff=0.000000, coherent_beam=0)
    beam_out = light_source.get_beam()


    if do_plot:
        beam = beam_out.duplicate()
        plot_scatter(beam.get_column(1) * 1e6, beam.get_column(3) * 1e6, xrange=[-1.1e3, 1.1e3], yrange=[-1.1e3, 1.1e3],
                     title="Real space")
    return beam_out


def source(amp=10,do_plot=1):
    a = SourceGridPolar(
        real_space=[2e-3, 0.0, 2e-3],
        real_space_points=[3, 8],
        real_space_center=[0.0, 0.0, 0.0],
        direction_space=[amp * 2e-3, amp * 2e-3],
        direction_space_points=[3, 359],
        direction_space_center=[0.0, 0.0])

    print(a.info())

    beam_out = a.get_beam()

    if do_plot:
        beam = beam_out.duplicate()
        plot_scatter(beam.get_column(1) * 1e6, beam.get_column(3) * 1e6,
                     title="Real space")
        plot_scatter(beam.get_column(4) * 1e6, beam.get_column(6) * 1e6,
                     title="Directions space")
        # plot_scatter(beam.get_column(1) * 1e6, beam.get_column(4) * 1e6,
        #              title="Phase space X")
        # plot_scatter(beam.get_column(3) * 1e6, beam.get_column(6) * 1e6,
        #              title="Phase space Z")


    return beam_out

def trace_ellipsoid(beam0, do_plot=0):
    #
    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
    optical_element = S4EllipsoidMirror(name='Ellipsoid Mirror', boundary_shape=None,
                                        surface_calculation=0, is_cylinder=0, cylinder_direction=0,
                                        convexity=1, min_axis=0.000000, maj_axis=0.000000, p_focus=2.100000,
                                        q_focus=1.050000,
                                        grazing_angle=0.113446,
                                        f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=2.1, q=1.05, angle_radial=1.45735)
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirrorElement
    beamline_element = S4EllipsoidMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                                input_beam=beam0)

    beam, mirr = beamline_element.trace_beam()

    if do_plot:
        from srxraylib.plot.gol import plot_scatter
        rays = beam.get_rays()
        plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='ELLIPSOID (X,Z) in microns')

    return beam

def trace_two_ellipsoids(beam0, theta2=83.5, do_plot=0):
    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
    optical_element = S4EllipsoidMirror(name='Ellipsoid Mirror', boundary_shape=None,
                                        surface_calculation=0, is_cylinder=0, cylinder_direction=0,
                                        convexity=1, min_axis=0.000000, maj_axis=0.000000, p_focus=2.100000,
                                        q_focus=3.150000,
                                        grazing_angle=0.113446,
                                        f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=2.1, q=3.15, angle_radial=1.45735)
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirrorElement
    beamline_element = S4EllipsoidMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                                input_beam=beam0)

    beam, mirr = beamline_element.trace_beam()

    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
    optical_element = S4EllipsoidMirror(name='Ellipsoid Mirror', boundary_shape=None,
                                        surface_calculation=0, is_cylinder=0, cylinder_direction=0,
                                        convexity=1, min_axis=0.000000, maj_axis=0.000000, p_focus=1.500000,
                                        q_focus=0.500000,
                                        grazing_angle=numpy.radians(90-theta2),
                                        f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=1.5, q=0.5, angle_radial=numpy.radians(theta2), angle_azimuthal=numpy.radians(180))
    from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirrorElement
    beamline_element = S4EllipsoidMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                                input_beam=beam)

    beam, mirr = beamline_element.trace_beam()


    if do_plot:
        from srxraylib.plot.gol import plot_scatter
        rays = beam.get_rays()
        plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='ELLIPSOID (X,Z) in microns')

    return beam


def trace_toroid(beam0, do_plot=0):
    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirror
    optical_element = S4ToroidalMirror(name='Toroidal Mirror', boundary_shape=None,
                                       surface_calculation=0,
                                       min_radius=0.100000, maj_radius=1.000000,
                                       p_focus=2.100000, q_focus=1.050000, grazing_angle=0.113446,
                                       f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=2.1, q=1.05, angle_radial=1.45735)
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirrorElement
    beamline_element = S4ToroidalMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                               input_beam=beam0)

    beam, mirr = beamline_element.trace_beam()

    if do_plot:
        from srxraylib.plot.gol import plot_scatter
        rays = beam.get_rays()
        plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='TOROID (X,Z) in microns')

    return beam

def trace_two_toroids(beam, theta2=83.5, do_plot=0):
    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirror
    optical_element = S4ToroidalMirror(name='Toroidal Mirror', boundary_shape=None,
                                       surface_calculation=0,
                                       min_radius=0.100000, maj_radius = 1.000000,
                                                                         p_focus = 2.100000, q_focus = 3.150000, grazing_angle = 0.113446,
                                                                                                                                 f_reflec = 0, f_refl = 0, file_refl = '<none>', refraction_index = 0.99999 + 0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates = ElementCoordinates(p=2.1, q=3.15, angle_radial=1.45735)
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirrorElement
    beamline_element = S4ToroidalMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                               input_beam=beam)

    beam, mirr = beamline_element.trace_beam()

    # optical element number XX
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirror
    optical_element = S4ToroidalMirror(name='Toroidal Mirror',boundary_shape=None,
        surface_calculation=0,
        min_radius=0.100000,maj_radius=1.000000,
        p_focus=1.500000,q_focus=0.500000,grazing_angle=numpy.radians(90-theta2),
        f_reflec=0,f_refl=0,file_refl='<none>',refraction_index=0.99999+0.001j)

    from syned.beamline.element_coordinates import ElementCoordinates
    coordinates=ElementCoordinates(p=1.5,q=0.5,angle_radial=numpy.radians(theta2), angle_azimuthal=numpy.radians(180))
    from shadow4.beamline.optical_elements.mirrors.s4_toroidal_mirror import S4ToroidalMirrorElement
    beamline_element = S4ToroidalMirrorElement(optical_element=optical_element,coordinates=coordinates,input_beam=beam)

    beam, mirr = beamline_element.trace_beam()


    # test plot
    if do_plot:
       from srxraylib.plot.gol import plot_scatter
       rays = beam.get_rays()
       plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='(X,Z) in microns')

    return beam

if __name__ == "__main__":
    # simulates Figs 1 to 6 from:

    # D.E.Aspnes, S.M.Kelso, "Properties And Performance Of Grazing Incidence Reflectors,
    # Proc. SPIE 0315, Reflecting Optics for Synchrotron Radiation, (3 May 1982);
    # doi: 10.1117 / 12.932985

    # beam0 = source(amp=10.0, do_plot=1)
    # beam = trace_ellipsoid(beam0, do_plot=1)


    # beam0 = source(amp=3.0, do_plot=0)
    # beam = trace_toroid(beam0, do_plot=1)

    # beam0 = source(amp=20.0, do_plot=1)
    # beam = trace_two_ellipsoids(beam0, do_plot=1)

    # beam0 = source(amp=3.0, do_plot=1)
    # beam = trace_two_toroids(beam0, do_plot=1)

    # beam0 = source(amp=50.0, do_plot=1)
    # beam = trace_two_ellipsoids(beam0, theta2=79.57, do_plot=1)

    beam0 = source(amp=3.0, do_plot=1)
    beam = trace_two_toroids(beam0, theta2=79.57, do_plot=1)