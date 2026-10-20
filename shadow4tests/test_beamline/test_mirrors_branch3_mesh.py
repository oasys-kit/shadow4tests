def check_congruence(oe):

    assert (oe.FHIT_C == 1)
    assert (oe.FSHAPE == 1)
    assert (oe.F_REFLEC == 0)
    assert (oe.FMIRR == 5)
    assert (oe.F_GRATING == 0)
    assert (oe.F_CRYSTAL == 0)
    assert (oe.F_G_S == 2)
    assert (oe.F_RIPPLE == 1)


if __name__ == "__main__":
    import numpy


    from shadow4.tools.graphics import plotxy

    from shadow4.beamline.optical_elements.mirrors.s4_surface_data_mirror import S4SurfaceDataMirror, \
        S4SurfaceDataMirrorElement
    from shadow4.syned.shape import Rectangle
    from shadow4.syned.element_coordinates import ElementCoordinates
    from shadow4.beam.beam import Beam

    #
    #
    #


    #
    # shadow3
    #
    from shadow4tests.oasys_workspaces.mirrors_branch3_mesh import define_source, run_source, define_beamline, run_beamline

    oe0 = define_source()
    beam3_source = run_source(oe0)



    #
    # shadow4
    #



    oe = define_beamline()[0]

    beam4_source = Beam.initialize_from_array(beam3_source.rays)
    beam4 = beam4_source


    check_congruence(oe)


    #
    # shadow definitions
    #


    name = "SurfaceDataMirror"
    mirror1 = S4SurfaceDataMirrorElement(optical_element=S4SurfaceDataMirror(
                                            name=name,
                                            surface_data_file="../oasys_workspaces/mirrors_branch3_mesh.hdf5",
                                            boundary_shape=Rectangle(
                                                        x_left=-oe.RWIDX2,
                                                        x_right=oe.RWIDX1,
                                                        y_bottom=-oe.RLEN2,
                                                        y_top=oe.RLEN1,
                                                                    )
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
    beam3 = run_beamline(beam3_source, define_beamline())

    plotxy(beam3, 1, 3, nbins=201, nolost=1, title="%s shadow3" % name)
    plotxy(beam4, 1, 3, nbins=201, nolost=1, title="%s shadow4" % name)

    from shadow4tests.compatibility.compare_beams import check_six_columns_mean_and_std, check_almost_equal

    check_six_columns_mean_and_std(beam3, beam4, do_assert=True, do_plot=False)
    check_almost_equal(beam3, beam4, do_assert = True, level=2)

