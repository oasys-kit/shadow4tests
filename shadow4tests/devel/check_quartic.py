import numpy
from srxraylib.profiles.diaboloid.fqs import single_quartic, single_quartic_modified, quartic_roots

def vy(ABCDE):
    a = ABCDE[0] #+ 0j
    b = ABCDE[1] #+ 0j
    c = ABCDE[2] #+ 0j
    d = ABCDE[3] #+ 0j
    e = ABCDE[4] + 0j
    if a!=1: raise Exception("a must be one.")

    D1 = 2 * c**3 - 9 * b * c * d + 27 * b**2 * e + 27 * d**2 - 72 * c * e
    D0 = c**2 - 3 * b * d + 12 * e
    D = (D1**2 - 4 * D0**3) / (-27)


    k = (8 * c - 3 * b**2) / 8


    Q = numpy.power(2, -1.0/3) * numpy.power((D1 + numpy.sqrt(D1**2 - 4 * D0**3)), 1.0/3)
    S = 0.5 * numpy.sqrt((1.0/3) * (Q + D0 / Q) - 2 * k / 3)
    if numpy.abs(S) < 1e-3:
        print(">>>> changed sign of sqrt")
        Q = numpy.power(2, -1.0/3) * numpy.power((D1 - numpy.sqrt(D1**2 - 4 * D0**3)), 1.0/3)
        S = 0.5 * numpy.sqrt((1.0/3) * (Q + D0 / Q) - 2 * k / 3)


    m = (b**3 - 4 * b * c + 8 * d) / 8

    print(">>>>>>>>>Q,S,D: ", Q,S,D)

    z1 = -b / 4 - S + 0.5 * numpy.sqrt(-4 * S**2 - 2 * k + m / S)
    z2 = -b / 4 - S - 0.5 * numpy.sqrt(-4 * S**2 - 2 * k + m / S)
    z3 = -b / 4 + S + 0.5 * numpy.sqrt(-4 * S**2 - 2 * k - m / S)
    z4 = -b / 4 + S - 0.5 * numpy.sqrt(-4 * S**2 - 2 * k - m / S)

    return z1, z2, z3, z4




def mquartic(a, b, c, d):
    from numpy import sqrt as Sqrt

    a += 0j
    b += 0j
    c += 0j
    d += 0j

    sol1 = -a/4. - Sqrt(a**2/4. - (2*b)/3. + (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) +  \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))/2. - Sqrt(a**2/2. - (4*b)/3. - (2**(1/3)*(b**2 - 3*a*c + 12*d)) /  \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) - \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3) / \
        (3.*2**(1/3)) - (-a**3 + 4*a*b - 8*c)/ \
        (4.*Sqrt(a**2/4. - (2*b)/3. + (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3) / \
        (3.*2**(1/3)))))/2.
    
    sol2 = -a/4. - Sqrt(a**2/4. - (2*b)/3. + \
        (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))/2. + Sqrt(a**2/2. - (4*b)/3. - (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) - \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)) - (-a**3 + 4*a*b - 8*c)/ \
        (4.*Sqrt(a**2/4. - (2*b)/3. + (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))))/2.

    sol3 = -a/4. + Sqrt(a**2/4. - (2*b)/3. + \
        (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))/2. - Sqrt(a**2/2. - (4*b)/3. - (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) - \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)) + (-a**3 + 4*a*b - 8*c)/ \
        (4.*Sqrt(a**2/4. - (2*b)/3. + (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))))/2.

    sol4 = -a/4. + Sqrt(a**2/4. - (2*b)/3. + \
        (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))/2. + Sqrt(a**2/2. - (4*b)/3. - (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) - \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)) + (-a**3 + 4*a*b - 8*c)/ \
        (4.*Sqrt(a**2/4. - (2*b)/3. + (2**(1/3)*(b**2 - 3*a*c + 12*d))/ \
        (3.*(2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)) + \
        (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d + Sqrt(-4*(b**2 - 3*a*c + 12*d)**3 + (2*b**3 - 9*a*b*c + 27*c**2 + 27*a**2*d - 72*b*d)**2))**(1/3)/ \
        (3.*2**(1/3)))))/2.


    return sol1, sol2, sol3, sol4

def newton(f,Df,x0,epsilon,max_iter):
    # '''Approximate solution of f(x)=0 by Newton's method.
    #
    # Parameters
    # ----------
    # f : function
    #     Function for which we are searching for a solution f(x)=0.
    # Df : function
    #     Derivative of f(x).
    # x0 : number
    #     Initial guess for a solution f(x)=0.
    # epsilon : number
    #     Stopping criteria is abs(f(x)) < epsilon.
    # max_iter : integer
    #     Maximum number of iterations of Newton's method.
    #
    # Returns
    # -------
    # xn : number
    #     Implement Newton's method: compute the linear approximation
    #     of f(x) at xn and find x intercept by the formula
    #         x = xn - f(xn)/Df(xn)
    #     Continue until abs(f(xn)) < epsilon and return xn.
    #     If Df(xn) == 0, return None. If the number of iterations
    #     exceeds max_iter, then return None.
    #
    # Examples
    # --------
    # >>> f = lambda x: x**2 - x - 1
    # >>> Df = lambda x: 2*x - 1
    # >>> newton(f,Df,1,1e-8,10)
    # Found solution after 5 iterations.
    # 1.618033988749989
    # '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if numpy.abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def pol4(z0, ABCDE=None):
    return ABCDE[0] * z0 ** 4 + ABCDE[1] * z0 ** 3 + ABCDE[2] * z0 ** 2 + ABCDE[3] * z0 + ABCDE[4]

def dpol4(z0, ABCDE=None):
    return 4 * ABCDE[0] * z0 ** 3 + 3 * ABCDE[1] * z0 ** 2 + 2 * ABCDE[2] * z0  + ABCDE[3]

if __name__ == "__main__":

    DCBA1 = [-8745.02188873291, -624.6460852022283, 149.91291022442874, -10.002177259318255, 1.0]
    print("\n>> coeffs DCBA1: ", DCBA1)

    # https://mathworld.wolfram.com/QuarticFormula.html


    if False:
        import numpy
        h_output2 = numpy.polynomial.polynomial.polyroots([DCBA1[0], DCBA1[1], DCBA1[2], DCBA1[3], DCBA1[4]])
        print("roots: ", h_output2)
        z = h_output2[0] ; print("np: 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0] )
        z = h_output2[1] ; print("np: 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0] )
        z = h_output2[2] ; print("np: 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0] )
        z = h_output2[3] ; print("np: 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0] )


    if False:

        # out = single_quartic(1.0, AA[k], BB[k], CC[k], DD[k])
        print(DCBA1[4], DCBA1[3], DCBA1[2], DCBA1[1], DCBA1[0])
        h_output = single_quartic(DCBA1[4], DCBA1[3], DCBA1[2], DCBA1[1], DCBA1[0])

        print(">>>> solutions1: ", h_output)
        z = h_output[0];print("single_quartic 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0])
        z = h_output[1];print("single_quartic 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0])
        z = h_output[2];print("single_quartic 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0])
        z = h_output[3];print("single_quartic 0 =? ", DCBA1[4] * z ** 4 + DCBA1[3] * z ** 3 + DCBA1[2] * z ** 2 + DCBA1[1] * z + DCBA1[0])



    if True:
        # ABCDE = [1, 7, -806, -1050, 38322]
        ABCDE = [ 1, -10.002177259318255, 149.91291022442874, -624.6460852022283, -8745.02188873291]
        ABCDE = [1, -32.73663672556182, 408.4190027490258, -2299.701822526753, 4891.1455078125]
        roots = quartic_roots(ABCDE, modified=1, zero_below=1e-6)
        print("roots: ", roots)
        # z0 = roots[0][0] ; print("modified: 0 =? ", ABCDE[0] * z0 ** 4 + ABCDE[1] * z0 ** 3 + ABCDE[2] * z0 ** 2 + ABCDE[3] * z0 + ABCDE[4])
        # z1 = roots[0][1] ; print("modified: 0 =? ", ABCDE[0] * z1 ** 4 + ABCDE[1] * z1 ** 3 + ABCDE[2] * z1 ** 2 + ABCDE[3] * z1 + ABCDE[4])
        # z2 = roots[0][2] ; print("modified: 0 =? ", ABCDE[0] * z2 ** 4 + ABCDE[1] * z2 ** 3 + ABCDE[2] * z2 ** 2 + ABCDE[3] * z2 + ABCDE[4])
        # z3 = roots[0][3] ; print("modified: 0 =? ", ABCDE[0] * z3 ** 4 + ABCDE[1] * z3 ** 3 + ABCDE[2] * z3 ** 2 + ABCDE[3] * z3 + ABCDE[4])

        z0 = roots[0][0] ; print("modified: 0 =? ", pol4(z0, ABCDE=ABCDE))
        z1 = roots[0][1] ; print("modified: 0 =? ", pol4(z1, ABCDE=ABCDE))
        z2 = roots[0][2] ; print("modified: 0 =? ", pol4(z2, ABCDE=ABCDE))
        z3 = roots[0][3] ; print("modified: 0 =? ", pol4(z3, ABCDE=ABCDE))




    #
        # # DCBA1 =   [-8745.02188873291, -624.6460852022283, 149.91291022442874, -10.002177259318255, 1.0]
        # >>> roots
            # array([[-30.76994812-0.j,  -7.60101564+0.j,   6.61999319+0.j,
            #          24.75097057-0.j]])

            # >>> roots = quartic_roots([1, 2, 3, 4, 5])
            # >>> roots
            # array([[-1.28781548-0.85789676j, -1.28781548+0.85789676j,
            #          0.28781548+1.41609308j,  0.28781548-1.41609308j]])
            #
            # >>> roots = quartic_roots([[1, 2, 3, 4, 5],
            #                            [1, 7, -806, -1050, 38322]])
            # >>> roots
            # array([[ -1.28781548-0.85789676j,  -1.28781548+0.85789676j,
            #           0.28781548+1.41609308j,   0.28781548-1.41609308j],
            #        [-30.76994812-0.j        ,  -7.60101564+0.j        ,
            #           6.61999319+0.j        ,  24.75097057-0.j        ]])
            #

        # print(mquartic(ABCDE[1], ABCDE[2], ABCDE[3], ABCDE[4]))
        # print(mquartic(7, -806, -1050, 38322))

    if True:
        # p = lambda x: x ** 3 - x ** 2 - 1
        # Dp = lambda x: 3 * x ** 2 - 2 * x
        # approx = newton(p, Dp, 1, 1e-10, 10)
        # print(approx)
        ABCDE = [1, -32.73663672556182, 408.4190027490258, -2299.701822526753, 4891.1455078125]
        p = lambda x: pol4(x, ABCDE=ABCDE)
        Dp = lambda x: dpol4(x, ABCDE=ABCDE)

        approx = newton(p, Dp, 10, 1e-10, 10)
        print(approx)
        z3 = roots[0][3] ; print("Newton: 0 =? ", pol4(approx, ABCDE=ABCDE))

        # approx = newton(p, Dp, -5, 1e-10, 10)
        # print(approx)
        # z3 = roots[0][3] ; print("Newton: 0 =? ", pol4(approx, ABCDE=ABCDE))


    if False:
        # ABCDE = [1, 7, -806, -1050, 38322]
        ABCDE = [ 1, -10.002177259318255, 149.91291022442874, -624.6460852022283, -8745.02188873291]
        ABCDE = [1, -32.73663672556182, 408.4190027490258, -2299.701822526753, 4891.1455078125]
        roots = vy(ABCDE)
        print("roots: ", roots)

        z0 = roots[0] ; print("vy modified: 0 =? ", pol4(z0, ABCDE=ABCDE))
        z1 = roots[1] ; print("vy modified: 0 =? ", pol4(z1, ABCDE=ABCDE))
        z2 = roots[2] ; print("vy modified: 0 =? ", pol4(z2, ABCDE=ABCDE))
        z3 = roots[3] ; print("vy modified: 0 =? ", pol4(z3, ABCDE=ABCDE))

