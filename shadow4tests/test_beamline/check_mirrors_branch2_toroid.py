def check_congruence(oe):

    assert (oe.FHIT_C == 0)
    assert (oe.F_REFLEC == 0)
    assert (oe.FMIRR == 3)
    assert (oe.F_GRATING == 0)
    assert (oe.F_CRYSTAL == 0)


if __name__ == "__main__":
    import numpy

    from srxraylib.plot.gol import set_qt


    set_qt()

    from shadow4.tools.graphics import plotxy

    from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import S4ToroidMirror, S4ToroidMirrorElement

    from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation
    from shadow4.beam.s4_beam import S4Beam as Beam

    #
    #
    #


    #
    # shadow3
    #
    from shadow4tests.oasys_workspaces.mirrors_branch2_toroid import define_source, run_source, define_beamline, run_beamline

    oe0 = define_source()
    beam3_source = run_source(oe0)



    #
    # shadow4
    #

    from syned.beamline.element_coordinates import ElementCoordinates

    oe_list = define_beamline() # just in case... reinitializa to "before run"
    oe = oe_list[0]

    beam4_source = Beam.initialize_from_array(beam3_source.rays)
    beam4 = beam4_source


    check_congruence(oe)


    #
    # shadow definitions
    #

    if oe.F_DEFAULT == 0:
        p_focus = oe.SSOUR
        q_focus = oe.SIMAG
        grazing_angle = numpy.radians(90 - oe.THETA)
    elif oe.F_DEFAULT == 1:
        p_focus = oe.T_SOURCE
        q_focus = oe.T_IMAGE
        grazing_angle = numpy.radians(90 - oe.T_INCIDENCE)

    name = "Toroid Mirror"

    ########################
    from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import S4ToroidMirror
    print("Grazing angle: ", grazing_angle)
    optical_element = S4ToroidMirror(name='Toroid Mirror', boundary_shape=None,
                                     surface_calculation=0,
                                     min_radius=0, maj_radius=0,
                                     p_focus=p_focus, q_focus=q_focus, grazing_angle=grazing_angle,
                                     f_reflec=0, f_refl=0, file_refl='<none>', refraction_index=0.99999 + 0.001j,
                                     coating_material='Si', coating_density=2.33, coating_roughness=0)

    from syned.beamline.element_coordinates import ElementCoordinates

    coordinates = ElementCoordinates(p=10, q=6, angle_radial=numpy.pi/2 - grazing_angle,
                                     angle_azimuthal=0, angle_radial_out=numpy.pi/2 - grazing_angle)
    movements = None
    from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import S4ToroidMirrorElement

    beamline_element = S4ToroidMirrorElement(optical_element=optical_element, coordinates=coordinates,
                                             movements=movements, input_beam=beam4_source)

    beam4, mirr4 = beamline_element.trace_beam()

    # ###################################################
    # mirror1 = S4ToroidMirrorElement(
    #     optical_element=S4ToroidMirror(
    #             name=name,
    #             boundary_shape=None,
    #             surface_calculation=SurfaceCalculation.INTERNAL,
    #             min_radius=0.0,
    #             maj_radius=0.0,
    #             p_focus=p_focus,
    #             q_focus=q_focus,
    #             grazing_angle=grazing_angle,
    #
    #         # inputs related to mirror reflectivity
    #             f_reflec=oe.F_REFLEC,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
    #             # f_refl=0,  # 0=prerefl file, 1=electric susceptibility, 2=user defined file (1D reflectivity vs angle)
    #             #             # 3=user defined file (1D reflectivity vs energy), # 4=user defined file (2D reflectivity vs energy and angle)
    #             # file_refl="",  # preprocessor file fir f_refl=0,2,3,4
    #             # refraction_index=1.0  # refraction index (complex) for f_refl=1
    #             ),
    #     coordinates=ElementCoordinates(
    #             p=oe.T_SOURCE,
    #             q=oe.T_IMAGE,
    #             angle_radial=numpy.radians(oe.T_INCIDENCE),
    #             ),
    # )
    #
    # print(mirror1.info())
    #
    # #
    # # run
    # #
    # print(">>>>>>>", beam4)
    # beam4, mirr4 = mirror1.trace_beam(beam_in=beam4, flag_lost_value=-11000)



    #
    # compare
    #
    oe_list = define_beamline()
    beam3 = run_beamline(beam3_source, oe_list)

    plotxy(beam3, 1, 3, nbins=201, nolost=1, title="%s shadow3" % name)
    plotxy(beam4, 1, 3, nbins=201, nolost=1, title="%s shadow4" % name)

    from shadow4tests.compatibility.compare_beams import check_six_columns_mean_and_std, check_almost_equal

    check_six_columns_mean_and_std(beam3, beam4, do_assert=True, do_plot=False)
    check_almost_equal(beam3, beam4, do_assert = True, level=3)

