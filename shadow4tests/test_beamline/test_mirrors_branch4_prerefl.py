def check_congruence(oe):

    assert (oe.FHIT_C == 1)
    assert (oe.FSHAPE == 1)
    assert (oe.F_REFLEC == 1)
    assert (oe.FMIRR == 5)
    assert (oe.F_GRATING == 0)
    assert (oe.F_CRYSTAL == 0)


if __name__ == "__main__":
    import numpy


    from shadow4.tools.graphics import plotxy

    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
    from shadow4.beam.beam import Beam

    #
    #
    #

    #
    # shadow3
    #
    from shadow4tests.oasys_workspaces.mirrors_branch4_prerefl import define_source, run_source, define_beamline, run_beamline

    oe0 = define_source()
    beam3_source = run_source(oe0)

    #
    # shadow4
    #

    from shadow4.syned.shape import Rectangle
    from shadow4.syned.element_coordinates import ElementCoordinates

    oe = define_beamline()[0]

    beam4_source = Beam.initialize_from_array(beam3_source.rays)
    beam4 = beam4_source

    check_congruence(oe)

    #
    # shadow definitions
    #
    name = "Plane Mirror with reflectivity"
    mirror1 = S4PlaneMirrorElement(
        optical_element=S4PlaneMirror(
                name=name,
                boundary_shape=Rectangle(
                    x_left=-oe.RWIDX2,
                    x_right=oe.RWIDX1,
                    y_bottom=-oe.RLEN2,
                    y_top=oe.RLEN1,
                    ),
                # inputs related to mirror reflectivity
                f_reflec=oe.F_REFLEC,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                f_refl=0,  # 0=prerefl file, 1=electric susceptibility, 2=user defined file (1D reflectivity vs angle)
                           # 3=user defined file (1D reflectivity vs energy), # 4=user defined file (2D reflectivity vs energy and angle)
                file_refl=oe.FILE_REFL,  # preprocessor file fir f_refl=0,2,3,4
                refraction_index=1.0  # refraction index (complex) for f_refl=1
                ),
        coordinates=ElementCoordinates(
                p=oe.T_SOURCE,
                q=oe.T_IMAGE,
                angle_radial=numpy.radians(oe.T_INCIDENCE),
                ),
    )

    print(mirror1.info())

    #
    # run
    #

    beam4, mirr4 = mirror1.trace_beam(beam_in=beam4, flag_lost_value=-11000)



    #
    # compare
    #
    oe_list = define_beamline()
    beam3 = run_beamline(beam3_source, oe_list)

    plotxy(beam3, 1, 3, nbins=201, nolost=1, title="%s shadow3" % name)
    plotxy(beam4, 1, 3, nbins=201, nolost=1, title="%s shadow4" % name)

    from shadow4tests.compatibility.compare_beams import check_six_columns_mean_and_std, check_almost_equal

    check_six_columns_mean_and_std(beam3, beam4, do_assert=True, do_plot=False)
    check_almost_equal(beam3, beam4, do_assert = True, level=2)