import numpy

#
#
#


#
# efields
#

def beam_get_efields(beam1, nolost=0):
    rays = beam1.get_rays(nolost=nolost)
    Es = rays[:,6:9].copy()
    Ep = rays[:,15:18].copy()
    phis = rays[:,13].copy()
    phip = rays[:,14].copy()
    return Es, phis, Ep, phip

def rotate_efield(E, theta1=0.0, axis=2):
        if axis == 1:
            torot = [2,3]
        elif axis == 2:
            torot = [1,3]
        elif axis == 3:
            torot = [1,2]


        costh = numpy.cos(theta1)
        sinth = numpy.sin(theta1)

        tstart = numpy.array([1,4,7,16])

        Enew = numpy.zeros_like(E)


        if axis == 1:
            newaxis = 0 # index
            newtorot = [1, 2]
        elif axis == 2:
            newaxis = 1
            newtorot = [0, 2]
        elif axis == 3:
            newaxis = 2
            newtorot = [0, 1]

        Enew[:,newtorot[0]] =  E[:,newtorot[0]] * costh + E[:,newtorot[1]] * sinth
        Enew[:,newtorot[1]] = -E[:,newtorot[0]] * sinth + E[:,newtorot[1]] * costh
        Enew[:,newaxis]     =  E[:,newaxis]

        return Enew

def beam_get_angle_Es_surface(beam1, n=None, nolost=0):
    from shadow4.tools.arrayofvectors import vector_modulus, vector_norm, vector_cross, vector_dot

    rays = beam1.get_rays(nolost=nolost)
    v = rays[:,3:6].copy()
    Es = rays[:,6:9].copy()
    Ep = rays[:,15:18].copy()

    if n is None:
        n = numpy.zeros_like(v)
        n[:, 2] = 1 # normal along z

    modEs = vector_modulus(Es)
    mask =  modEs > 1e-10

    # P is a unit vector on the surface (perpendicular to both v and n)
    P = vector_norm(vector_cross(v, n))

    # print(">>>> mask:", mask)
    # print(">>>> v: ", v)
    # print(">>>> |v|: ", vector_modulus(v))
    # print(">>>> n: ", n)
    # print(">>>> P=v x n: ", P)


    # angle with Es

    if mask.all():
        # print(">>>> OK with Es")
        # print(">>>> Es=: ", vector_norm(Es))
        cos_alpha1 = vector_dot(vector_norm(Es), P)
        alpha = numpy.arccos(cos_alpha1)
    else:
        # print("NOT OK with Es")
        # print(">>>> Ep[~mask]=: ", vector_norm(Ep[~mask]))
        alpha = numpy.zeros(modEs.size)
        alpha[mask] = numpy.arccos(vector_dot(vector_norm(Es[mask]), P[mask]))
        alpha[~mask] = numpy.arccos(vector_dot(vector_norm(Ep[~mask]), P[~mask])) + numpy.pi / 2

    return alpha
#
# jones
#

# defines a Jones vector (for npoints)
def jones(j1, j2):
    j = numpy.zeros((j1.size, 2), dtype=complex)
    j[:, 0] = j1
    j[:, 1] = j2
    return j

# defines a Jones matric (a tuple)
def Jones(a11=0.0, a12=0.0, a21=0.0, a22=0.0):
    return (a11, a12, a21, a22)

# matrix multiplication J x j
def JXj(J, j1):
    a11, a12, a21, a22 = J
    j2 = numpy.zeros_like(j1, dtype=complex)
    j2[:, 0] = a11 * j1[:, 0] + a12 * j1[:, 1]
    j2[:, 1] = a21 * j1[:, 0] + a22 * j1[:, 1]
    return j2

# matrix multiplication J1 x 2
def JXJ(J1, J2):
    a11, a12, a21, a22 = J1
    b11, b12, b21, b22 = J2
    c11 = a11 * b11 + a12 * b21
    c12 = a11 * b12 + a12 * b22
    c21 = a21 * b11 + a22 * b21
    c22 = a21 * b12 + a22 * b22
    return (c11, c12, c21, c22)

# matrix rotation R(-alpha) J R(alpha)
def Jones_rotated(J1, alpha=0):
    return JXJ(Rotation(-alpha), JXJ(J1, Rotation(alpha)))

# rotation matrix
def Rotation(alpha):
        c = numpy.cos(alpha)
        s = numpy.sin(alpha)
        return (c, s, -s, c)

def Jones_diagonal_rotated(alpha, J=(1,0,0,1)):
    a11, a12, a21, a22 = J
    if a12 != 0.0: raise Exception("J[0]=a12 must be zero")
    if a21 != 0.0: raise Exception("J[2]=a21 must be zero")
    c = numpy.cos(alpha)
    s = numpy.sin(alpha)
    return (a11 * c**2 + a22 * s**2, (a11 - a22) * s * c, (a11 - a22) * s * c, a11 * s**2 + a22 * c**2)

# i/o
def jones_from_beam(beam1, nolost=0):
    rays = beam1.get_rays(nolost=nolost)
    Js = rays[:,[6,8]].copy().astype(complex)
    Js[:, 0] = Js[:, 0] * numpy.exp(1j * rays[:,13])
    Js[:, 1] = Js[:, 1] * numpy.exp(1j * rays[:,13])

    Jp = rays[:,[15,17]].copy().astype(complex)
    Jp[:, 0] = Jp[:, 0] * numpy.exp(1j * rays[:,14])
    Jp[:, 1] = Jp[:, 1] * numpy.exp(1j * rays[:,14])
    return Js + Jp

def jones_to_efields(J, reset_phis=0):
    n = J.shape[0]
    Es = numpy.zeros((n, 3), dtype=float)
    Ep = numpy.zeros((n, 3), dtype=float)

    Es[:, 0] = numpy.abs(J[:, 0])
    Ep[:, 2] = numpy.abs(J[:, 1])
    phis = numpy.angle(J[:, 0])
    phip = numpy.angle(J[:, 1])

    if reset_phis:
        PP = phis.copy()
        phis -= PP
        phip -= PP
        Es = rotate_efield(Es, theta1=PP, axis=2)
        Ep = rotate_efield(Ep, theta1=PP, axis=2)


    return Es, phis, Ep, phip



#
# misc
#
def get_source(polarization_degree=0.5, phase_diff=numpy.pi/2):
    #
    #
    #
    from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
    light_source = SourceGeometrical(name='SourceGeometrical', nrays=3, seed=5676561)
    light_source.set_spatial_type_point()
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_flat(hdiv1=0.000000, hdiv2=0.000000, vdiv1=0.000000, vdiv2=0.000000)
    light_source.set_energy_distribution_singleline(1000.000000, unit='eV')
    light_source.set_polarization(polarization_degree=polarization_degree, phase_diff=phase_diff, coherent_beam=1)
    beam = light_source.get_beam()

    # test plot
    # from srxraylib.plot.gol import plot_scatter
    # rays = beam.get_rays()
    # plot_scatter(1e6 * rays[:, 0], 1e6 * rays[:, 2], title='(X,Z) in microns')
    return beam



# def jones_apply_matrix(J1, a11=1.0, a12=0.0, a21=0.0, a22=1.0):
#     J2 = numpy.zeros_like(J1, dtype=complex)
#     J2[:, 0] = J1[:, 0] * a11 + J1[:, 1] * a12
#     J2[:, 1] = J1[:, 0] * a21 + J1[:, 1] * a22
#     return J1


def get_case(
        polarization_degree=1.0,
        phase_diff=0.0,
        beam_rotation=0.0,
        rs=1.0,
        rp=0.0,
        alpha=0.0,
        ):

    B = get_source(polarization_degree=polarization_degree, phase_diff=phase_diff)
    B.rotate(beam_rotation, axis=2)

    Es, phis, Ep, phip = beam_get_efields(B)
    print("Es, phis: ", Es, numpy.degrees(phis))
    print("Ep, phip: ", Ep, numpy.degrees(phip))

    j = jones_from_beam(B)
    print("original j:", j)
    J = Jones(rs, 0, 0, rp)
    Jrot = Jones_diagonal_rotated(alpha, J)
    print("Jrot: ", Jrot)
    j1 = JXj(Jrot, j)
    print("reflected j:", j1)
    print("converted to efields:")
    EEs, pphis, EEp, pphip = jones_to_efields(j1)
    print("reflected EEs, pphis: ", EEs, numpy.degrees(pphis))
    print("reflected EEp, pphip: ", EEp, numpy.degrees(pphip))

    return Es, phis, Ep, phip, EEs, pphis, EEp, pphip


if __name__ == "__main__":
    # from shadow4.beam.s4_beam import S4Beam
    # B = S4Beam.initialize_as_pencil(N=10)
    # Es, phis, Ep, phip = beam_get_efields(B)
    # print(Es, phis)
    # print(Ep, phip)

    B = get_source(polarization_degree=0.5, phase_diff=numpy.pi/2)
    B.rotate(numpy.pi/4, axis=2)
    # print(B.efields_orthogonal())
    # print(B.intensity())

    print("\n\n>>>>>>> EFIELDS <<<<<<<<<")
    Es, phis, Ep, phip = beam_get_efields(B)
    print(Es, numpy.degrees(phis))
    print(Ep, numpy.degrees(phip))
    print("rotated Es 45 deg: ", rotate_efield(Es, theta1=numpy.pi/4, axis=2))

    print("\n\n>>>>>>> JONES VECTORS <<<<<<<<<")
    j = jones_from_beam(B)
    print(j)
    print("j times the Identity", JXj( Jones(1, 0, 0, 1), j))
    print("converted to efields:")
    EEs, pphis, EEp, pphip = jones_to_efields(j)
    print(EEs, numpy.degrees(pphis))
    print(EEp, numpy.degrees(pphip))

    print("converted to efields with reset:")
    EEs, pphis, EEp, pphip = jones_to_efields(j, reset_phis=1)
    print(EEs, numpy.degrees(pphis))
    print(EEp, numpy.degrees(pphip))

    print("\n\n>>>>>>> JONES MATRICES <<<<<<<<<")
    rs = 0.9 + 0.09j
    rp = 0.1 + 0.01j
    Jrot0 = Jones_diagonal_rotated(0, J=(rs, 0, 0, rp))
    Jrot45 = Jones_diagonal_rotated(numpy.pi / 4, J=(rs, 0, 0, rp))
    Jrot0new = Jones_rotated((rs, 0, 0, rp), 0.0)
    Jrot45new = Jones_rotated((rs, 0, 0, rp), numpy.pi / 4)
    print("Jrot0: ", Jrot0)
    print("Jrot45: ", Jrot45)
    print("Jrot0new: ", Jrot0new)
    print("Jrot45new: ", Jrot45new)
    print("JXj=", JXj(Jrot0, j))

    #case 1
    if False:
        print("\n\n>>>>>>> CASE 1 <<<<<<<<<")

        phase_diff = numpy.pi / 2
        polarization_degree = 0.5
        beam_rotation = 0.0
        alpha = 0
        rs = 0.9 + 0.9j
        rp = 0.1 + 0.1j

        Es, phis, Ep, phip, EEs, pphis, EEp, pphip = get_case(phase_diff=phase_diff,
                                                              polarization_degree=polarization_degree,
                                                              beam_rotation=beam_rotation,
                                                              alpha=alpha,
                                                              rs=rs,
                                                              rp=rp,
                                                              )

        print(">>>> hand calculations: ")
        print("reflected, s: ", Es * numpy.abs(rs), numpy.degrees(phis + numpy.angle(rs) ))
        print("reflected, p: ", Ep * numpy.abs(rp), numpy.degrees(phip + numpy.angle(rp) ))


    #case 2
    if False:
        print("\n\n>>>>>>> CASE 2 <<<<<<<<<")

        phase_diff = numpy.pi / 2
        polarization_degree = 0.5
        beam_rotation = 0.0
        alpha = numpy.pi / 2
        rs = 0.9 + 0.9j
        rp = 0.1 + 0.1j

        Es, phis, Ep, phip, EEs, pphis, EEp, pphip = get_case(phase_diff=phase_diff,
                                                              polarization_degree=polarization_degree,
                                                              beam_rotation=beam_rotation,
                                                              alpha=alpha,
                                                              rs=rs,
                                                              rp=rp,
                                                              )

        print(">>>> hand calculations: ")
        print("reflected, s: ", Es * numpy.abs(rp), numpy.degrees(phis + numpy.angle(rp) ))
        print("reflected, p: ", Ep * numpy.abs(rs), numpy.degrees(phip + numpy.angle(rs) ))


    #case 3
    if False:
        print("\n\n>>>>>>> CASE 3 <<<<<<<<<")

        phase_diff = numpy.pi / 2
        polarization_degree = 0.5
        beam_rotation = numpy.pi / 2
        alpha = 0
        rs = 0.9 + 0.9j
        rp = 0.1 + 0.1j

        Es, phis, Ep, phip, EEs, pphis, EEp, pphip = get_case(phase_diff=phase_diff,
                                                              polarization_degree=polarization_degree,
                                                              beam_rotation=beam_rotation,
                                                              alpha=alpha,
                                                              rs=rs,
                                                              rp=rp,
                                                              )

        print(">>>> hand calculations: ")
        print("reflected, s: ", Ep * numpy.abs(rs), numpy.degrees(phip + numpy.angle(rs) ))
        print("reflected, p: ", Es * numpy.abs(rp), numpy.degrees(phis + numpy.angle(rp) ))

    #case 4
    if False:
        print("\n\n>>>>>>> CASE 4 <<<<<<<<<")

        phase_diff = numpy.pi / 2
        polarization_degree = 0.5
        beam_rotation = numpy.pi / 2
        alpha = numpy.pi / 2
        rs = 0.9 + 0.9j
        rp = 0.1 + 0.1j

        Es, phis, Ep, phip, EEs, pphis, EEp, pphip = get_case(phase_diff=phase_diff,
                                                              polarization_degree=polarization_degree,
                                                              beam_rotation=beam_rotation,
                                                              alpha=alpha,
                                                              rs=rs,
                                                              rp=rp,
                                                              )

        print(">>>> hand calculations: ")
        print("reflected, s: ", Ep * numpy.abs(rp), numpy.degrees(phip + numpy.angle(rp) ))
        print("reflected, p: ", Es * numpy.abs(rs), numpy.degrees(phis + numpy.angle(rp) ))



    B = get_source(polarization_degree=0.5, phase_diff=numpy.pi/2)
    B.rays[0, 6:9] = 0.0
    B.rays[1:4, 15:18] = 0.0
    # print(">>>> B.rays[:, 6:9]", B.rays[:, 6:9])
    B.rotate(numpy.pi / 2, axis=2)

    #
    # put beam in mirror reference system
    #
    alpha1 = 0.0
    theta_grazing1 = numpy.pi / 2
    p = 10.0

    B.rotate(alpha1, axis=2)
    B.rotate(theta_grazing1, axis=1)
    B.translation([0.0, -p * numpy.cos(theta_grazing1), p * numpy.sin(theta_grazing1)])

    alphas = beam_get_angle_Es_surface(B)
    alphas[0] = 0
    print("alphas: ", numpy.degrees(alphas))


    Jr = Jones_rotated(Jones(1,2,3,4), alphas)
    print(Jr)