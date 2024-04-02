import numpy

from srxraylib.plot.gol import set_qt

set_qt()

from shadow4.tools.graphics import plotxy

from shadow4.beamline.optical_elements.mirrors.s4_paraboloid_mirror import S4ParaboloidMirror, S4ParaboloidMirrorElement
from syned.beamline.shape import Convexity, Direction, Side

from shadow4.beamline.s4_optical_element_decorators import SurfaceCalculation

from shadow4.beam.s4_beam import S4Beam as Beam



def check_congruence(oe):

    assert (oe.FHIT_C == 0)
    assert (oe.F_REFLEC == 0)
    assert (oe.FMIRR == 4)
    assert (oe.F_GRATING == 0)
    assert (oe.F_CRYSTAL == 0)


def run_ellipsoid(kind="paraboloid"):
    #
    # shadow3
    #
    if kind == "paraboloid":
        from shadow4tests.oasys_workspaces.mirrors_branch5_paraboloid import define_source, run_source, define_beamline, run_beamline
    elif kind == "paraboloid_tangential_cylinder":
        from shadow4tests.oasys_workspaces.mirrors_branch5_paraboloid_tangential_cylinder import define_source, run_source, define_beamline, run_beamline
    else:
        raise Exception("Bad input")

    oe0 = define_source()
    beam3_source = run_source(oe0)



    #
    # shadow4
    #

    from syned.beamline.element_coordinates import ElementCoordinates

    oe = define_beamline()[0]

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


    is_cylinder = oe.FCYL

    if oe.CIL_ANG == 0:
        cylinder_direction = Direction.TANGENTIAL
    else:
        cylinder_direction = Direction.SAGITTAL

    if oe.F_CONVEX == 0:
        convexity = Convexity.DOWNWARD
    elif oe.F_CONVEX == 1:
        convexity = Convexity.UPWARD

    # f_side = 0 - for fmirr=4: focus location at image (0) or source (1).

    if oe.F_SIDE == 0:
        at_infinity = Side.IMAGE
    elif oe.F_SIDE == 1:
        at_infinity = Side.SOURCE



    name = "Paraboloid Mirror (%s) " % kind
    mirror1 = S4ParaboloidMirrorElement(
        optical_element=S4ParaboloidMirror(
                name=name,
                boundary_shape=None,
                surface_calculation=SurfaceCalculation.INTERNAL,
                is_cylinder=is_cylinder,
                cylinder_direction=cylinder_direction,
                convexity=convexity,
                parabola_parameter=0.0,
                at_infinity=at_infinity,
                pole_to_focus=0.0,
                p_focus=p_focus,
                q_focus=q_focus,
                grazing_angle=grazing_angle,

            # inputs related to mirror reflectivity
                f_reflec=oe.F_REFLEC,  # reflectivity of surface: 0=no reflectivity, 1=full polarization
                # f_refl=0,  # 0=prerefl file, 1=electric susceptibility, 2=user defined file (1D reflectivity vs angle)
                #             # 3=user defined file (1D reflectivity vs energy), # 4=user defined file (2D reflectivity vs energy and angle)
                # file_refl="",  # preprocessor file fir f_refl=0,2,3,4
                # refraction_index=1.0  # refraction index (complex) for f_refl=1
                ),
        coordinates=ElementCoordinates(
                p=oe.T_SOURCE,
                q=oe.T_IMAGE,
                angle_radial=numpy.radians(oe.T_INCIDENCE),
                ),
        input_beam=beam4,
    )

    print(mirror1.to_python_code())

    #
    # run
    #

    beam4, mirr4 = mirror1.trace_beam()



    #
    # compare
    #
    oe_list = define_beamline()
    beam3 = run_beamline(beam3_source, oe_list)

    plotxy(beam3, 1, 3, nbins=201, nolost=1, title="%s shadow3" % name)
    plotxy(beam4, 1, 3, nbins=201, nolost=1, title="%s shadow4" % name)

    from shadow4tests.compatibility.compare_beams import check_six_columns_mean_and_std, check_almost_equal

    check_six_columns_mean_and_std(beam3, beam4, do_assert=True, do_plot=False, assert_value=1e-6)
    check_almost_equal(beam3, beam4, do_assert = True, level=2)


if __name__ == "__main__":

    run_ellipsoid("paraboloid")
    run_ellipsoid("paraboloid_tangential_cylinder")
