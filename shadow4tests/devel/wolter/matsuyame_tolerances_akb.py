#
# Python script to run shadow3. Created automatically with make_python_script_from_list().
#
import Shadow
import numpy

import numpy as np
def rotate_and_translate_coefficients(coe_list,R_M,T):
    axx, ayy, azz, axy, ayz, axz, ax, ay, az, a0 = coe_list
    A2 = np.array([[axx,axy/2,axz/2],
    [axy/2,ayy,ayz/2],
    [axz/2,ayz/2,azz]])
    A1 = np.array([ax,ay,az])
    A0 = a0
    B2 = np.dot(R_M, np.dot(A2,R_M.T)) # first equation 6.29
    B1 = np.dot(R_M, A1) - 2 * np.dot(B2,T) # 2nd equation 6.29
    B0 = A0 + np.dot(T.T, (np.dot(B2, T) - \
    np.dot(R_M, A1))) # 3rd equation 6.29
    return [ B2[0,0], B2[1,1], B2[2,2], B2[0,1] + B2[1,0], B2[1,2] + B2[2,1], B2[0,2] + B2[2,0], B1[0], B1[1], B1[2], B0]

def define_source():
    #
    # initialize shadow3 source (oe0) and beam
    #
    oe0 = Shadow.Source()

    # Define variables. See https://raw.githubusercontent.com/oasys-kit/shadow3/master/docs/source.nml

    oe0.CONE_MAX = 5e-06
    oe0.FDISTR = 5
    oe0.FSOUR = 2
    oe0.F_PHOT = 0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = 5676561
    oe0.NPOINT = 50000
    oe0.PH1 = 11500.0
    oe0.WXSOU = 1.25e-05
    oe0.WZSOU = 1.25e-05

    return oe0


def run_source(oe0, iwrite=False):
    # iwrite (1) or not (0) SHADOW files start.xx end.xx star.xx

    # Run SHADOW to create the source

    if iwrite:
        oe0.write("start.00")

    beam = Shadow.Beam()
    beam.genSource(oe0)

    if iwrite:
        oe0.write("end.00")
        beam.write("begin.dat")

    return beam


def define_beamline(X_ROT=0.0, X_ROT2=0.0,
                    ccc=[0.0, -46.20118343205854, -1847960.8414621525, 0.0, -27444.927571197237, 0.0, 0.0, 1.1510792319313623e-11,
         2985.178365734318, 0.0]):

    # initialize elements
    oe_list = []

    oe1 = Shadow.OE()
    oe_list.append(oe1)
    oe2 = Shadow.OE()
    oe_list.append(oe2)
    oe3 = Shadow.OE()
    oe_list.append(oe3)
    oe4 = Shadow.OE()
    oe_list.append(oe4)
    oe5 = Shadow.OE()
    oe_list.append(oe5)
    oe6 = Shadow.OE()
    oe_list.append(oe6)

    # Define variables. See https://raw.githubusercontent.com/oasys-kit/shadow3/master/docs/oe.nml

    oe1.CCC = numpy.array(
        [0.0, 0.025865678682546505, 3299.1783677260473, 0.0, 17.783187372192028, 0.0, 0.0, 0.0, -31.151332442494663,
         0.0])
    oe1.DUMMY = 100.0
    oe1.FHIT_C = 1
    oe1.FMIRR = 10
    oe1.FWRITE = 1
    oe1.F_EXT = 1
    oe1.RLEN1 = 0.065
    oe1.RLEN2 = 0.065
    oe1.RWIDX1 = 0.025
    oe1.RWIDX2 = 0.025
    oe1.T_IMAGE = 0.085
    oe1.T_INCIDENCE = 89.83957181736336
    oe1.T_REFLECTION = 89.83957181736336
    oe1.T_SOURCE = 45.0

    oe2.ALPHA = 90.0
    oe2.CCC = numpy.array(
        [0.0, 0.045836362869578186, 5846.4462174678365, 0.0, 32.04741572531343, 0.0, 0.0, 0.0, -31.291997277978908,
         0.0])
    oe2.DUMMY = 100.0
    oe2.FHIT_C = 1
    oe2.FMIRR = 10
    oe2.FWRITE = 1
    oe2.F_EXT = 1
    oe2.F_MOVE = 1
    oe2.RLEN1 = 0.065
    oe2.RLEN2 = 0.065
    oe2.RWIDX1 = 0.025
    oe2.RWIDX2 = 0.025
    oe2.T_IMAGE = 0.05
    oe2.T_INCIDENCE = 89.83957181736336
    oe2.T_REFLECTION = 89.83957181736336
    oe2.T_SOURCE = 0.085
    oe2.X_ROT = X_ROT

    oe3.ALPHA = 270.0
    oe3.DUMMY = 100.0
    oe3.FWRITE = 3
    oe3.F_REFRAC = 2
    oe3.T_IMAGE = 0.0
    oe3.T_INCIDENCE = 0.0
    oe3.T_REFLECTION = 180.0
    oe3.T_SOURCE = 0.0

    oe4.CCC = numpy.array(
        [0.0, -14.759924385621062, -590369.3425480056, 0.0, -8767.850207060355, 0.0, 0.0, -2.3501200985265314e-12,
         1687.2747284606646, 0.0])
    oe4.DUMMY = 100.0
    oe4.FHIT_C = 1
    oe4.FMIRR = 10
    oe4.FWRITE = 1
    oe4.F_EXT = 1
    oe4.RLEN1 = 0.015
    oe4.RLEN2 = 0.015
    oe4.RWIDX1 = 0.05
    oe4.RWIDX2 = 0.05
    oe4.T_IMAGE = 0.025
    oe4.T_INCIDENCE = 89.71352110243458
    oe4.T_REFLECTION = 89.71352110243458
    oe4.T_SOURCE = 0.05


    oe5.ALPHA = 90.0
    oe5.CCC = numpy.array(ccc
        )
    oe5.DUMMY = 100.0
    oe5.FHIT_C = 1
    oe5.FMIRR = 10
    oe5.FWRITE = 1
    oe5.F_EXT = 1
    oe5.RLEN1 = 0.015
    oe5.RLEN2 = 0.015
    oe5.RWIDX1 = 0.05
    oe5.RWIDX2 = 0.05
    oe5.T_IMAGE = 0.065
    oe5.T_INCIDENCE = 89.71352110243458
    oe5.T_REFLECTION = 89.71352110243458
    oe5.T_SOURCE = 0.025
    oe5.F_EXT = 1
    oe5.F_MOVE = 1
    oe5.X_ROT = X_ROT2

    oe6.ALPHA = 270.0
    oe6.DUMMY = 100.0
    oe6.FWRITE = 3
    oe6.F_REFRAC = 2
    oe6.T_IMAGE = 0.0
    oe6.T_INCIDENCE = 0.0
    oe6.T_REFLECTION = 180.0
    oe6.T_SOURCE = 0.0

    return oe_list


def run_beamline(beam_in, oe_list, iwrite=0):


    beam = beam_in.duplicate()

    #
    # run optical element 1
    #
    print("    Running optical element: %d" % (1))
    oe1 = oe_list[1 - 1]
    if iwrite:
        oe1.write("start.01")

    beam.traceOE(oe1, 1)
    oe1 = oe_list[1 - 1]
    if iwrite:
        oe1.write("end.01")
        beam.write("star.01")

    #
    # run optical element 2
    #
    print("    Running optical element: %d" % (2))
    oe2 = oe_list[2 - 1]
    if iwrite:
        oe2.write("start.02")

    beam.traceOE(oe2, 2)
    oe2 = oe_list[2 - 1]
    if iwrite:
        oe2.write("end.02")
        beam.write("star.02")

    #
    # run optical element 3
    #
    print("    Running optical element: %d" % (3))
    oe3 = oe_list[3 - 1]
    if iwrite:
        oe3.write("start.03")

    beam.traceOE(oe3, 3)
    oe3 = oe_list[3 - 1]
    if iwrite:
        oe3.write("end.03")
        beam.write("star.03")

    #
    # run optical element 4
    #
    print("    Running optical element: %d" % (4))
    oe4 = oe_list[4 - 1]
    if iwrite:
        oe4.write("start.04")

    beam.traceOE(oe4, 4)
    oe4 = oe_list[4 - 1]
    if iwrite:
        oe4.write("end.04")
        beam.write("star.04")

    #
    # run optical element 5
    #
    print("    Running optical element: %d" % (5))
    oe5 = oe_list[5 - 1]
    if iwrite:
        oe5.write("start.05")

    beam.traceOE(oe5, 5)
    oe5 = oe_list[5 - 1]
    if iwrite:
        oe5.write("end.05")
        beam.write("star.05")

    #
    # run optical element 6
    #
    print("    Running optical element: %d" % (6))
    oe6 = oe_list[6 - 1]
    if iwrite:
        oe6.write("start.06")

    beam.traceOE(oe6, 6)
    oe6 = oe_list[6 - 1]
    if iwrite:
        oe6.write("end.06")
        beam.write("star.06")

    return beam


#
# main
#
if __name__ == "__main__":
    from srxraylib.plot.gol import set_qt
    set_qt()

    method = 1 # 0=shadow, 1=rotation of coefficients

    oe0 = define_source()

    beam0 = run_source(oe0, iwrite=0)

    tol_a_n = 55
    # tol_a = numpy.linspace(-150e-6, 150e-6, tol_a_n)
    # tol_a_deg = numpy.degrees(tol_a) #
    
    tol_a_deg = numpy.linspace(-0.001, 0.001, tol_a_n) # numpy.degrees(tol_a) #
    tol_a = numpy.radians(tol_a_deg)

    fwhm = numpy.zeros_like(tol_a_deg)

    ccc0 = [0.0, -46.20118343205854, -1847960.8414621525, 0.0, -27444.927571197237, 0.0, 0.0, 1.1510792319313623e-11,
     2985.178365734318, 0.0]

    for i in range(tol_a_n):
        beam = beam0.duplicate()



        if method == 0:
            oe_list = define_beamline(X_ROT2=tol_a_deg[i])
        else:
            ######################################
            Theta = tol_a[i]
            R_M = numpy.array([[1, 0, 0],
                            [0, numpy.cos(Theta), -numpy.sin(Theta)],
                            [0, numpy.sin(Theta),  numpy.cos(Theta)]])
            # translation vector
            T = np.array([0, 0, 0])

            ccc1 = rotate_and_translate_coefficients(ccc0.copy(),R_M,T)
            #####################################
            oe_list = define_beamline(ccc=ccc1) # X_ROT2=tol_a_deg[i])




        beam = run_beamline(beam, oe_list, iwrite=0) #tol_a[i])

        # Shadow.ShadowTools.plotxy(beam, 1, 3, nbins=101, nolost=1, title="Real space %f (rad)" % (tol_a[i]))
        width = 5000e-9
        nbins = 501

        tkt = beam.histo1(1, nbins=nbins, nolost=1, xrange=[-width,width])
        fwhm[i] = tkt['fwhm']
        print(">>>> FWHM [nm]: ", tkt['fwhm'])
        if tol_a_n < 10: Shadow.ShadowTools.histo1(beam, 1, nbins=nbins, nolost=1, xrange=[-width, width])

        # print(tkt)
        # tkt = beam.histo1(1, )
    # Shadow.ShadowTools.plotxy(beam,1,4,nbins=101,nolost=1,title="Phase space X")
    # Shadow.ShadowTools.plotxy(beam,3,6,nbins=101,nolost=1,title="Phase space Z")
    from srxraylib.plot.gol import plot
    plot(1e6*tol_a, 1e9*fwhm, xtitle="X_ROT [urad]", ytitle="FWHM [nm]")
    filename = "matsuyama_tolerances_akb3.dat"
    f = open(filename,'w')
    for i in range(tol_a_n):
        print(i, 1e6*tol_a[i], 1e9 * fwhm[i])
        f.write("%g   %g  \n" % (1e6*tol_a[i], 1e9*fwhm[i]))
    f.close()
    print("File %s written to disk." % filename)

    a = numpy.loadtxt(filename)
    plot(a[:,0], a[:,1])