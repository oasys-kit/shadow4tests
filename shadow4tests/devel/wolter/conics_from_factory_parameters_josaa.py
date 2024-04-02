
from shadow4.optical_surfaces.s4_conic import S4Conic # for comparison
from shadow4tests.devel.wolter.conic_viewer import view_conic, compare_conics # for plot

import numpy

from numpy import sin as Sin
from numpy import cos as Cos
from numpy import tan as Tan
from numpy import sqrt as Sqrt

def Cot(x):
    return 1/Tan(x)

def Sec(x):
    return 1/Cos(x)

def Csc(x):
    return 1/Sin(x)


def normalize(c_in, index=0, clean=True):
    c_out = [0] * 10
    for i in range(10):
        c_out[i] = c_in[i] / c_in[index]
        if clean:
            if numpy.abs(c_out[i]) < 1e-15:
                c_out[i] = 0.0
    return c_out


#
# parabolas
#


# see conics_penelope_paraboloid_focusing.nb
def paraboloid_ken(p=1e10, q=10,theta=3e-3):
    if p <=0 or q <=0: raise Exception("Error: p,q must be strictly positives")
    s = numpy.sin(theta)
    c = numpy.cos(theta)

    if p > q: # using q
        out = [1.,
               s ** 2,
               c ** 2,
               0,
               2 * s * c,
               0,
               0,
               0,
               -4 * s * q,
               0,
               ]
    else: # using p
        out = [1.,
               s ** 2,
               c ** 2,
               0,
               -2 * s * c,
               0,
               0,
               0,
               -4 * s * p,
               0,
               ]

    return out

def paraboloid_s4(p=10,q=3,theta=3e-3, normalized=0):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    ccc = S4Conic.initialize_as_paraboloid_from_focal_distances(p, q, theta,
                                        cylindrical=0, cylangle=0.0, switch_convexity=0)

    if normalized: out = normalize(ccc.get_coefficients(), index=0, clean=1)
    return out

def paraboloid_josaa(p=1e10, q=10,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    s = numpy.sin(theta)
    c = numpy.cos(theta)

    if p > q: # using q
        out = [1.,
               s ** 2,
               c ** 2,
               0,
               2 * s * c,
               0,
               0,
               0,
               -4 * s * q,
               0,
               ]
    else: # using p
        out = [1.,
               s ** 2,
               c ** 2,
               0,
               -2 * s * c,
               0,
               0,
               0,
               -4 * s * p,
               0,
               ]

    return out

#
# ellipsoid
#
# see conics_penelope_ellipsoid.nb
def ellipsoid(p=10,q=3,theta=3e-3, normalized=0):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")

    out =  [Csc(theta)**2/(p*q),
            1/(p*q),
            (-(p - q)**2 + (p + q)**2*Csc(theta)**2)/(p*q*(p + q)**2),
            0,
    (2*(p - q)*Sqrt(((p + q)**2*Cos(theta)**2)/(p**2 + q**2 + 2*p*q*Cos(2*theta)))*Sqrt(p**2 + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt(Csc(theta)**2/(p + q)**2))/(p*q*(p + q)),
            0,
            0,
    (4*(p - q)*(-(Sqrt(p*q)*Sqrt((p*q*Cos(theta)**2)/(p**2 + q**2 + 2*p*q*Cos(2*theta)))*Csc(theta)) +
        p*q*Sqrt(((p + q)**2*Cos(theta)**2)/(p**2 + q**2 + 2*p*q*Cos(2*theta)))*Sqrt(Csc(theta)**2/(p + q)**2)))/
        (p*q*(p + q)*Sqrt(p**2 + q**2 + 2*p*q*Cos(2*theta))*Sqrt(Csc(theta)**2/(p + q)**2)),
    -(Sqrt(Csc(theta)**2/(p + q)**2)*(-2*(p**2 - q**2)**2*Cot(theta)**2 +
    Csc(theta)**2*((p - q)**2*(p**2 + 6*p*q + q**2) + (p - q)**4*Cos(2*theta) +
       (8*(p*q)**1.5*Cos(theta)*Sqrt(((p + q)**2*Cos(theta)**2)/(p**2 + q**2 + 2*p*q*Cos(2*theta)))*Cot(theta))/
        (Sqrt((p*q*Cos(theta)**2)/(p**2 + q**2 + 2*p*q*Cos(2*theta)))*Sqrt(Csc(theta)**2/(p + q)**2))))*Sin(theta)**2)/
    (2.*p*q*(p**2 + q**2 + 2*p*q*Cos(2*theta))),
            0]

    if normalized: out = normalize(out, index=0, clean=1)
    return out

def ellipsoid_ken(p=3,q=10,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    c = Cos(theta)
    s = Sin(theta)
    h = (p - q) * c
    out = [
        1,
        s**2,
        (h**2 + 4 * p * q) / (p + q)**2,
        0,
        -2 * s * c  * (q-p) / (q+p),
        0,
        0,
        0,
        -4 * s * p * q / (p + q),
        0
    ]

    return out

def ellipsoid_s4(p=10,q=3,theta=3e-3, normalized=0):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    ccc = S4Conic.initialize_as_ellipsoid_from_focal_distances(p, q, theta,
                                        cylindrical=0, cylangle=0.0, switch_convexity=0)

    if normalized: out = normalize(ccc.get_coefficients(), index=0, clean=1)
    return out

def ellipsoid_josaa(p=10,q=3,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    s = numpy.sin(theta)
    a = 0.5 * (p + q)
    b = numpy.sqrt(p * q) * s
    c = numpy.sqrt(a**2 - b**2)

    center_y = (p**2 - q**2) / (4 * c)
    # center_z = -b * numpy.sqrt(1 - (center_y / a)**2)  # bad sign!!!
    center_z = - p * q * numpy.sin(2 * theta) / 2 / c
    # center_z = - b ** 2 / (2 * c * numpy.sin(theta))

    normal_y = -2 * center_y / a**2
    normal_z = -2 * center_z / b**2
    normal_mod = numpy.sqrt(normal_y**2 + normal_z**2)
    normal_y /= normal_mod
    normal_z /= normal_mod

    # print(">>>>>> a,b,c:", a, b, c)
    # print(">>>>>> center: 0,", center_y, center_z)
    # print(">>>>>> normal: 0,", normal_y, normal_z)

    B_over_A = b**2 / a**2

    out = [1,
           normal_y ** 2 + B_over_A * normal_z ** 2,
           normal_z ** 2 + B_over_A * normal_y ** 2,
           0,
           2 * normal_y * normal_z *(B_over_A - 1),
           0,
           0,
           0,
           2 * (normal_z * center_z + B_over_A * normal_y * center_y),
           0,
           ]
    return out



#
# hyperboloid
#


def hyperboloid(p=10,q=3,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    if p >= q:
        return hyperboloid_large_p(p=p,q=q,theta=theta)
    else:
        return hyperboloid_large_q(p=p, q=q, theta=theta)

def hyperboloid_large_p(p=10,q=3,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    if p < q:
        raise Exception("p<q")
    return [
        -(Csc(theta)**2/(p*q)),-(((p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*Csc(theta)**2)/
        (p*q*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2))),
        (4*(p + q)**2 - ((p - q)**4*Csc(theta)**2*(1 + Csc(theta)**2))/(p*q))/((p - q)**2*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)),0,
        (-2*(p + q)*(p**2 + q**2 - 2*p*q*Cos(2*theta))*Csc(theta)**2*Sqrt(1/(1 + (p + q)**2/((p - q)**2*(1 + Csc(theta)**2)))))/
        (p*(p - q)**3*q*Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))),0,0,
        -2*((-2*(p + q)*Sqrt(1/(1 + (p + q)**2/((p - q)**2*(1 + Csc(theta)**2)))))/
        ((p - q)*Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))) +
        (Sqrt(2)*(-p**2 + q**2)*Sqrt(-((p*q*(-3 + Cos(2*theta)))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))*
        Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))*Sin(theta))/
        (Sqrt(p*q)*(-2*(p**2 - p*q + q**2) + (p**2 + q**2)*Cos(2*theta)))),
        -((((p + q)**2*(p**2 + q**2 - 2*p*q*Cos(2*theta))*Csc(theta)*Sqrt(1/(2 + (2*(p + q)**2)/((p - q)**2*(1 + Csc(theta)**2))))*
        (4*Sqrt(p*q)*Sqrt(-((p*q*(-3 + Cos(2*theta)))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta)))) +
        (p - q)**2*Sqrt(((p - q)**2*(-3 + Cos(2*theta)))/(-2*(p**2 - p*q + q**2) + (p**2 + q**2)*Cos(2*theta)))*Csc(theta)*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))))/(p*q) +
        2*(4*(p + q)**2 - ((p - q)**4*Csc(theta)**2*(1 + Csc(theta)**2))/(p*q))*
        ((p + q)**2/(2.*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))) -
        Sqrt(2)*Sqrt(p*q)*Sqrt(-((p*q*(-3 + Cos(2*theta)))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))*
        Sqrt(1/(1 + (p + q)**2/((p - q)**2*(1 + Csc(theta)**2))))*Sin(theta)))/
        ((p - q)**2*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2))),0]

def hyperboloid_large_q(p=3,q=10,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    if p > q:
        raise Exception("p>q")
    return [
        -(Csc(theta)**2/(p*q)),-(((p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*Csc(theta)**2)/
        (p*q*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2))),
        (4*(p + q)**2 - ((p - q)**4*Csc(theta)**2*(1 + Csc(theta)**2))/(p*q))/((p - q)**2*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)),0,
        (2*(p**2 + q**2 - 2*p*q*Cos(2*theta))*Sqrt(-((p*q*(-3 + Cos(2*theta)))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))*Csc(theta)**3*
        Sqrt((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)))/
        ((p - q)**2*(p*q)**1.5*Sqrt((4*(p**2 + q**2) + 2*(p - q)**2*Csc(theta)**2)/
        ((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))),0,0,
        -2*(-(Csc(theta)*(-4*p*q*Sqrt((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)) +
        ((p - q)*(p + q)*Csc(theta)**2)/
        (Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))))*
        (1 + Sin(theta)**2))/
        (2.*Sqrt(p*q)*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)*
        Sqrt((p*q*(1 + Sin(theta)**2))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta)))) +
        ((p**2 + q**2 - 2*p*q*Cos(2*theta))*Sqrt(-((p*q*(-3 + Cos(2*theta)))/(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))*Csc(theta)**3*
        Sqrt((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2))*
        ((-2*p*q*(-3 + Cos(2*theta)))/
        ((p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))) +
        ((p - q)*(p + q)*Sqrt(((p + q)**2*Sin(theta)**2)/((p - q)**2 + 2*(p**2 + q**2)*Sin(theta)**2)))/
        Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))/
        (2.*(p - q)**2*(p*q)**1.5*Sqrt((4*(p**2 + q**2) + 2*(p - q)**2*Csc(theta)**2)/
        ((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta)))))),
        -2*((2*(p + q)*((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2))**1.5)/
        ((p - q)*Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))) +
        ((p - q)*(-3 + Cos(2*theta))*(-2*(p**2 - p*q + q**2) + (p**2 + q**2)*Cos(2*theta))*Csc(theta)**4*
        (p*(Sqrt((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)) -
        Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))))) +
        q*(Sqrt((p + q)**2/(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)) +
        Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*
        Sqrt((2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)/((p - q)**2*(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta)))))))/
        (Sqrt(p**2 - 4*p*q + q**2 + 2*p*q*Cos(2*theta))*(2*(p**2 + q**2) + (p - q)**2*Csc(theta)**2)**2)),0]




def hyperboloid_ken(p=3,q=10,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    c = Cos(theta)
    s = Sin(theta)
    return [
        1,
        s**2,
        c**2 - 4 * p *q * s**2 / (q - p)**2,
        0,
        - 2 * s * c * (p + q) / (q - p),
        0,
        0,
        0,
        -4 * s * p * q  / (q - p),
        0
    ]

def hyperboloid_s4(p=10,q=3,theta=3e-3, normalized=0):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    ccc = S4Conic.initialize_as_hyperboloid_from_focal_distances(p, q, theta,
                                        cylindrical=0, cylangle=0.0, switch_convexity=0)

    if normalized: out = normalize(ccc.get_coefficients(), index=0, clean=1)
    return out


def hyperboloid_josaa(p=10,q=3,theta=3e-3):
    if p <= 0 or q <= 0: raise Exception("Error: p,q must be strictly positives")
    s = numpy.sin(theta)
    a = 0.5 * numpy.abs(p - q)
    c = 0.5 * numpy.sqrt(p**2 + q**2 - 2 * p * q * numpy.cos(2 * theta))
    b = numpy.sqrt(c**2 - a**2)

    if p > q:
        center_y = (p**2 - q**2) / (4 * c)
        center_z = b * numpy.sqrt(numpy.abs((center_y / a)**2 - 1))

        normal_y = -2 * center_y / a**2
        normal_z =  2 * center_z / b**2
    else:
        center_y = (p**2 - q**2) / (4 * c)
        center_z = b * numpy.sqrt(numpy.abs((center_y / a)**2 - 1))

        normal_y =  2 * center_y / a**2
        normal_z = -2 * center_z / b**2

    normal_mod = numpy.sqrt(normal_y ** 2 + normal_z ** 2)
    normal_y /= normal_mod
    normal_z /= normal_mod


    print(">>>>>> a,b,c:", a, b, c)
    print(">>>>>> center: 0,", center_y, center_z)
    print(">>>>>> normal: 0,", normal_y, normal_z)

    B_over_A = b**2 / a**2

    out = [1,
           normal_y ** 2 - B_over_A * normal_z ** 2,
           normal_z ** 2 - B_over_A * normal_y ** 2,
           0,
           2 * normal_y * normal_z *(-B_over_A - 1),
           0,
           0,
           0,
           2 * (normal_z * center_z - B_over_A * normal_y * center_y),
           0,
           ]
    return out

#
# tools
#
def cylinder(c_in):
    c_out = c_in.copy()
    c_out[0] = 0.0
    c_out[3] = 0.0
    c_out[5] = 0.0
    c_out[6] = 0.0
    return c_out



if __name__ == "__main__":

    if 0:
        p = 10**10
        q = 3
        theta = 3e-3

        ccc_ken = paraboloid_ken(p, q, theta)
        ccc_josaa = paraboloid_josaa(p, q, theta)
        ccc_s4 = paraboloid_s4(p, q, theta, normalized=1)

        print("parabola   ken: ", ccc_ken)
        print("parabola josaa: ", ccc_josaa)
        print("parabola    s4: ", ccc_s4)

    if 0:
        p = 3
        q = 1e10
        theta = 3e-3

        ccc_ken = paraboloid_ken(p, q, theta)
        ccc_josaa = paraboloid_josaa(p, q, theta)
        ccc_s4 = paraboloid_s4(p, q, theta, normalized=1)

        print("parabola   ken: ", ccc_ken)
        print("parabola josaa: ", ccc_josaa)
        print("parabola    s4: ", ccc_s4)


    if 0:
        p = 10
        q = 3
        theta = 3e-3

        ccc_ken = ellipsoid_ken(p, q, theta)
        ccc_josaa = ellipsoid_josaa(p, q, theta)
        ccc_s4 = ellipsoid_s4(p, q, theta, normalized=1)

        print("ellipse   ken: ", ccc_ken)
        print("ellipse josaa: ", ccc_josaa)
        print("ellipse    s4: ", ccc_s4)

        print("center_y: ", (p**2-q**2) / 4 / numpy.sqrt(0.25 * (p+q)**2 - p * q * numpy.sin(theta)**2))
        aa = (p**2-q**2)**2 / (4 * (p+q)**2 * (0.25 * (p+q)**2 - p *q * numpy.sin(theta)**2) )
        print("center_z: ", -numpy.sqrt(p * q) * numpy.sin(theta) * numpy.sqrt( 1 - aa))
        print("c_yy: ", p * q * numpy.sin(2 * theta)**2 / (p**2 + q**2 + 2 * p * q * numpy.cos(2 * theta)))

    if 0:
        p = 10
        q = 3
        theta = 3e-3

        ccc_ken =   hyperboloid_ken(p, q, theta)
        ccc_josaa = hyperboloid_josaa(p, q, theta)
        ccc_s4 =    hyperboloid_s4(p, q, theta, normalized=1)

        print("hyperbola   ken: ", ccc_ken)
        print("hyperbola josaa: ", ccc_josaa)
        print("hyperbola    s4: ", ccc_s4)


    #
    # scan
    #
    if 1:
        from srxraylib.plot.gol import plot

        P = numpy.linspace(0.1,9.9,100)
        C_josaa = numpy.zeros_like(P)
        C_ken = numpy.zeros_like(P)
        C_s4 = numpy.zeros_like(P)

        for index in range(0,10):
            for i,p in enumerate(P):
                print("\n\n p:", p)
                q = 10
                theta = 3e-3

                ccc_ken =   hyperboloid_ken(p, q, theta)
                ccc_josaa = hyperboloid_josaa(p, q, theta)
                ccc_s4 =    hyperboloid_s4(p, q, theta, normalized=1)
                C_josaa[i] = ccc_josaa[index]
                C_ken[i] = ccc_ken[index]
                C_s4[i] = ccc_s4[index]


            # if index == 1:
            #     for jjj in range(P.size):
            #         print(C_josaa[jjj], C_ken[jjj], C_s4[jjj])

            plot(P,C_s4, P,C_ken, P,C_josaa, legend=['s4','ken','josaa'], title="index = %d" % index)


        p = 10
        q = 3
        theta = 3e-3

        ccc_ken = ellipsoid_ken(p, q, theta)
        ccc_josaa = ellipsoid_josaa(p, q, theta)
        ccc_s4 = ellipsoid_s4(p, q, theta, normalized=1)

        print("ellipse   ken: ", ccc_ken)
        print("ellipse josaa: ", ccc_josaa)
        print("ellipse    s4: ", ccc_s4)

        print("center_y: ", (p**2-q**2) / 4 / numpy.sqrt(0.25 * (p+q)**2 - p * q * numpy.sin(theta)**2))
        aa = (p**2-q**2)**2 / (4 * (p+q)**2 * (0.25 * (p+q)**2 - p *q * numpy.sin(theta)**2) )
        print("center_z: ", -numpy.sqrt(p * q) * numpy.sin(theta) * numpy.sqrt( 1 - aa))
        print("c_yy: ", p * q * numpy.sin(2 * theta)**2 / (p**2 + q**2 + 2 * p * q * numpy.cos(2 * theta)))
    # print(paraboloid_focusing(q=10,theta=3e-3))
    # print(paraboloid_collimating(p=10, theta=3e-3))
    # print(hyperboloid_large_p(p=10, q=3, theta=3e-3))
    # print(hyperboloid_large_q(p=3, q=10, theta=3e-3))
    #
    #
    # parabola_check(ssour=10e10,simag=10,theta_grazing=3e-3, do_plot=False)
    # parabola_check(ssour=10, simag=10e10, theta_grazing=3e-3, do_plot=False)
    # hyperbola_check(ssour=10, simag=3, theta_grazing=3e-3, do_plot=False)
    # hyperbola_check(ssour=3, simag=10, theta_grazing=3e-3, do_plot=False)

    # print("cylinder, p<q:")
    # print(normalize(cylinder(hyperboloid_large_q(p=3, q=10, theta=3e-3)), index=2))
    # print(normalize(ken_hyperboloid_large_q_old(p=3, q=10, theta=3e-3),index=2))
    #
    # print("cylinder, p>q:")
    # print(normalize(cylinder(hyperboloid_large_p(p=10, q=3, theta=3e-3)), index=2))
    # print(normalize(ken_hyperboloid_large_p_old(p=10, q=3, theta=3e-3),index=2))

    # print("hyperboloid, p<q:")
    # p, q, theta = 7, 10, 3e-1
    # # p, q, theta = 0.900000, 2.700000, 0.003
    # print(normalize(hyperboloid_large_q(p=p, q=q, theta=theta), index=0))
    # print(ken_hyperboloid_large_q(p=p, q=q, theta=theta))
    # ccc = ken_hyperboloid_large_q(p=p, q=q, theta=theta)
    # s1, s2 = plot_height(ccc, p=p, q=q, theta=theta, title="p=%f, q=%f, theta=%f" % (p,q,theta))
    #
    # print("hyperboloid, p>q:")
    # p, q, theta = 10, 7, 3e-1
    # # p, q, theta = 2.7, 0.900000, 0.003
    # print(normalize(hyperboloid_large_p(p=10, q=7, theta=3e-3), index=0))
    # print(ken_hyperboloid_large_p(p=p, q=q, theta=theta))
    # ccc = ken_hyperboloid_large_p(p=p, q=q, theta=theta)
    # s1, s2 = plot_height(ccc, p=p, q=q, theta=theta, title="p=%f, q=%f, theta=%f" % (p,q,theta))



    # print("ellipsoid")
    # print(normalize(ellipsoid(p=10, q=3, theta=3e-3), index=0))
    # print(normalize(ken_ellipsoid(p=10, q=3, theta=3e-3), index=0))
    #
    # print("parabola")
    # print(normalize(paraboloid_focusing(q=10, theta=3e-3), index=0))
    # print(normalize(ken_paraboloid_focusing(q=10, theta=3e-3), index=0))