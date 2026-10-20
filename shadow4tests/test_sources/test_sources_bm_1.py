def check_congruence(oe0):
    assert(oe0.F_WIGGLER == 0)
    assert( (oe0.FDISTR == 4) or (oe0.FDISTR == 6))


if __name__ == "__main__":


    from syned.storage_ring.electron_beam import ElectronBeam
    from shadow4.sources.bending_magnet.s4_bending_magnet import S4BendingMagnet
    from shadow4.sources.bending_magnet.s4_bending_magnet_light_source import S4BendingMagnetLightSource

    from shadow4.tools.graphics import plotxy


    #
    #
    #

    to_meters = 1.0

    #
    # shadow3
    #
    from shadow4tests.oasys_workspaces.sources_bm_1 import define_source, run_source

    oe0 = define_source()
    beam3 = run_source(oe0)



    #
    # shadow4
    #

    check_congruence(oe0)

    electron_beam = ElectronBeam(energy_in_GeV=oe0.BENER,current=0.2,
                                       moment_xx   = (oe0.SIGMAX * to_meters)**2,
                                       moment_xpxp = (oe0.EPSI_X / oe0.SIGMAX)**2,
                                       moment_yy   = (oe0.SIGMAZ * to_meters)**2,
                                       moment_ypyp = (oe0.EPSI_Z / oe0.SIGMAZ)**2,
                                 )

    bm = S4BendingMagnet(
        radius=oe0.R_ALADDIN * to_meters,
        magnetic_field=S4BendingMagnet.calculate_magnetic_field(oe0.R_MAGNET * to_meters, oe0.BENER),
        length= (oe0.HDIV1 + oe0.HDIV2) * oe0.R_MAGNET,  # BM
        emin=oe0.PH1,  # Photon energy scan from energy (in eV)
        emax=oe0.PH2,  # Photon energy scan to energy (in eV)
        ng_e=200,  # Photon energy scan number of points
        ng_j=100,  # Number of points in electron trajectory (per period) for internal calculation only
        flag_emittance=True,  # when sampling rays: Use emittance (0=No, 1=Yes)
        )

    ls_bm = S4BendingMagnetLightSource(electron_beam=electron_beam,
                                       bending_magnet_magnetic_structure=bm)

    print(ls_bm.info())

    beam4 = ls_bm.get_beam(
        F_COHER=oe0.F_COHER,
        NRAYS=oe0.NPOINT,
        SEED=oe0.ISTAR1,
        EPSI_DX=oe0.EPSI_DX,
        EPSI_DZ=oe0.EPSI_DZ,
        psi_interval_in_units_one_over_gamma=None,
        psi_interval_number_of_points=1001,
        verbose=False,
    )


    #
    # compare
    #

    plotxy(beam3, 1, 3, nbins=201, title="BM shadow3")
    plotxy(beam4, 1, 3, nbins=201, title="BM shadow4")

    from shadow4tests.compatibility.compare_beams import check_six_columns_mean_and_std

    check_six_columns_mean_and_std(beam3, beam4, do_plot = True, do_assert = True, assert_value = 1e-2, to_meters=to_meters)

