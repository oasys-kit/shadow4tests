# This is similar to sync_f in srxraylib but faster
import numpy
from scipy.special import kv, gamma
# todo: move to sr-xraylib
def sync_f_sigma_and_pi(rAngle, rEnergy):
    r""" angular dependency of synchrotron radiation emission

      NAME:
            sync_f_sigma_and_pi

      PURPOSE:
            Calculates the function used for calculating the angular
         dependence of synchrotron radiation.

      CATEGORY:
            Mathematics.

      CALLING SEQUENCE:
            Result = sync_f_sigma_and_pi(rAngle,rEnergy)

      INPUTS:
            rAngle:  (array) the reduced angle, i.e., angle[rads]*Gamma. It can be a
             scalar or a vector.
            rEnergy:  (scalar) a value for the reduced photon energy, i.e.,
             energy/critical_energy.

      KEYWORD PARAMETERS:


      OUTPUTS:
            returns the value  of the sync_f for sigma and pi polarizations
             The result is an array of the same dimension as rAngle.

      PROCEDURE:
            The number of emitted photons versus vertical angle Psi is
         proportional to sync_f, which value is given by the formulas
         in the references.


         References:
             G K Green, "Spectra and optics of synchrotron radiation"
                 BNL 50522 report (1976)
             A A Sokolov and I M Ternov, Synchrotron Radiation,
                 Akademik-Verlag, Berlin, 1968

      OUTPUTS:
            returns the value  of the sync_f function

      PROCEDURE:
            Uses BeselK() function

      MODIFICATION HISTORY:
            Written by:     M. Sanchez del Rio, srio@esrf.fr, 2002-05-23
         2002-07-12 srio@esrf.fr adds circular polarization term for
             wavelength integrated spectrum (S&T formula 5.25)
         2012-02-08 srio@esrf.eu: python version
         2019-10-31 srio@lbl.gov  speed-up changes for shadow4

    """

    #
    # ; Eq 11 in Pag 6 in Green 1975
    #
    ji = numpy.sqrt((1.0 + rAngle**2)**3) * rEnergy / 2.0
    efe_sigma = kv(2.0 / 3.0, ji) * (1.0 + rAngle**2)
    efe_pi = rAngle * kv(1.0 / 3.0, ji) / numpy.sqrt(1.0 + rAngle ** 2) * (1.0 + rAngle ** 2)
    return efe_sigma**2, efe_pi**2

def kv_approx(nu, x):
    # https://www.researchgate.net/figure/Modified-Bessel-function-K-2-3-x-together-with-its-corresponding-fit-according_fig2_235356942
    # Research in Astron. Astrophys. 2013 Vol. 13 No. 6, 680–686
    # http://www.raa-journal.org http://www.iop.org/journals/raa

    if nu == 1/3:
        a1 = [-1.3746667760953621,
              0.44040512552162292,
              -0.15527012012316799]

        a2 = [-0.33550751062084]
    elif nu == 2/3:
        a1 = [-1.3746667760953621,
              0.44040512552162292,
              -0.15527012012316799]

        a2 = [-0.33550751062084]
        # a1 = [-1.0010216415582440,
        #         0.88350305221249859,
        #         3.6240174463901829 ,
        #         0.57393980442916881]
        #
        # a2 = [-0.2493940736333195,
        #     +0.9122693061687756,
        #     1.2051408667145216,
        #     -5.5227048291651126]
    elif nu == 5/3:
        a1 = [-1.0194198041210243,
            +0.28011396300530672,
            -7.71058491739234908e-2]

        a2 = [-15.761577796582387]

    a1 = numpy.array(a1)
    a2 = numpy.array(a2)

    H1 = 0
    H2 = 0

    for i in range(a1.size):
        H1 += a1[i] * x ** (1 / (i + 1))

    for i in range(a2.size):
        H2 += a2[i] * x ** (1 / (i + 1))

    delta1 = numpy.exp(H1) ################ minus!!
    delta2 = 1 - numpy.exp(H2)

    A1 = 0.5 * gamma(nu) * (x / 2)**(-nu)
    A2 = numpy.sqrt(numpy.pi / (2 * x)) * numpy.exp(-x)
    out =  A1 * delta1 + A2 * delta2
    return out

# def kv_approx_coeff(a12, x, nu = 2 / 3):
#     # https://goi.org/10.1088/1674-4527/13/6/007
#     # https://www.researchgate.net/figure/Modified-Bessel-function-K-2-3-x-together-with-its-corresponding-fit-according_fig2_235356942
#     # Research in Astron. Astrophys. 2013 Vol. 13 No. 6, 680–686
#     # http://www.raa-journal.org http://www.iop.org/journals/raa
#
#
#
#     a12 = numpy.array(a12)
#
#     H1 = 0
#     H2 = 0
#
#     for i in [0,1,2]:
#         H1 += a12[i] * x ** (1 / (i + 1))
#
#     H2 += a12[3] * x ** (1 / (0 + 1))
#
#     delta1 = numpy.exp(H1)
#     delta2 = 1 - numpy.exp(H2)
#
#     A1 = 0.5 * gamma(nu) * (x / 2)**(-nu)
#     A2 = numpy.sqrt(numpy.pi / (2 * x)) * numpy.exp(-x)
#     out =  A1 * delta1 + A2 * delta2
#     return out

def kv_approx_fine(nu, x):
    # Approximated expressions for the modified Bessel functions K1/3, K2/3 and K5/3
    # Coefficients have been fitted using the expression in:
    # https://goi.org/10.1088/1674-4527/13/6/007
    # See file shadow4-tests/shadow4tests/devel/fit_bessel_kv.py
    if numpy.abs(nu - 1/3) < 1e-10:
        coeffs = [-0.31902416, -0.81577317, -0.78202672,  0.30405889,  0.70028439, -1.16431336,
  0.24015406, -0.0261485 ]
    elif numpy.abs(nu - 2/3) < 1e-10:
        coeffs = [-0.37896593, -0.34095854, -0.62947205,  0.05467015,  0.62890735, -1.07260337,
  1.66367831, -1.78893917]
    elif numpy.abs(nu - 5/3) < 1e-10:
        coeffs = [-2.35033577e-01,  2.17241138e-02, -7.04622366e-03,  9.65554026e-04,
  7.64819524e-01, -4.54068899e+00,  1.11791188e+01, -7.25096908e+00]
    else:
        raise Exception("Fit coefficients not available for nu=%f" % nu)

    H1 = 0
    H2 = 0

    ii = 0
    for i in [0,1,2,3]:
        ii += 1
        H1 += coeffs[i] * x ** (1 / ii)

    ii = 0
    for i in [4,5,6,7]:
        ii += 1
        H2 += coeffs[i] * x ** (1 / ii)

    delta1 = numpy.exp(H1)
    delta2 = 1 - numpy.exp(H2)

    A1 = 0.5 * gamma(nu) * (x / 2)**(-nu)
    A2 = numpy.sqrt(numpy.pi / (2 * x)) * numpy.exp(-x)
    out =  A1 * delta1 + A2 * delta2
    return out


def kv_approx_coeff_fine(a12, x, nu = 2/3):
    # https://goi.org/10.1088/1674-4527/13/6/007
    # https://www.researchgate.net/figure/Modified-Bessel-function-K-2-3-x-together-with-its-corresponding-fit-according_fig2_235356942
    # Research in Astron. Astrophys. 2013 Vol. 13 No. 6, 680–686
    # http://www.raa-journal.org http://www.iop.org/journals/raa

    a12 = numpy.array(a12)

    H1 = 0
    H2 = 0

    ii = 0
    for i in [0,1,2,3]:
        ii += 1
        H1 += a12[i] * x ** (1 / ii)

    ii = 0
    for i in [4,5,6,7]:
        ii += 1
        H2 += a12[i] * x ** (1 / ii)

    delta1 = numpy.exp(H1)
    delta2 = 1 - numpy.exp(H2)

    A1 = 0.5 * gamma(nu) * (x / 2)**(-nu)
    A2 = numpy.sqrt(numpy.pi / (2 * x)) * numpy.exp(-x)
    out =  A1 * delta1 + A2 * delta2
    return out

def residual(params, nu=1/3):
    ji_interval_number_of_points = 1001
    x = numpy.linspace(01e-10, 5, ji_interval_number_of_points)
    return kv_approx_coeff_fine(params, x, nu=nu) - kv(nu,x)

if __name__ == "__main__":
    import time

    if False:
        psi1 = 0.0
        psi_interval_number_of_points = 1001
        angle_array_reduced = numpy.linspace(-0.5 * psi1, 0.5 * psi1, psi_interval_number_of_points)

        t0 = time.time()
        for i in range(5000):
            tmp = sync_f_sigma_and_pi(angle_array_reduced, 100.)
        print("Time: ", time.time()-t0)


    if False:

        ji_interval_number_of_points = 1001
        ji_array = numpy.linspace(0, 5, ji_interval_number_of_points)

        k13 = kv(1.0 / 3.0, ji_array)
        k23 = kv(2.0 / 3.0, ji_array)
        k53 = kv(5.0 / 3.0, ji_array)

        print(ji_array)
        approx = kv_approx(2/3, ji_array)
        print(approx)

        from srxraylib.plot.gol import plot

        plot(ji_array, k13,
             ji_array, k23,
             ji_array, k53,
             ji_array, approx,
             legend=['k1/3','k2/3','k5/3', 'k approx'], xlog=0, ylog=0, yrange=[1e-18,10])


    if False:

        # calculate fitting coefficients

        ji_interval_number_of_points = 1001
        ji_array = numpy.linspace(01e-10, 5, ji_interval_number_of_points)

        nu=2/3


        from scipy.optimize import least_squares

        # x0 = numpy.array([-1.3746667760953621  * 0.1     ,
        #       0.44040512552162292              * 0.1     ,
        #       -0.15527012012316799             * 0.1     ,
        #       -0.33550751062084                * 0.1    ])

        x0 = [
            -1.0010216415582440   * 0.1,
            +0.88350305221249859  * 0.1,
            -3.6240174463901829   * 0.1,
            +0.57393980442916881  * 0.1,
            -0.2493940736333195 * 0.1,
            +0.9122693061687756 * 0.1,
            +1.2051408667145216 * 0.1,
            -5.5227048291651126 * 0.1,
        ]
        res = least_squares(residual, x0, jac='2-point', method='lm', #bounds=(-inf, inf),
                      ftol=1e-08, xtol=1e-09, gtol=1e-09, x_scale=1.0, loss='linear', f_scale=1.0,
                      diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0,
                      args=(), kwargs={'nu':nu})

        print("Initial coeffs: ", x0)
        print("Final coeffs: ", res.x)
        # print("Correct coeffs: ", coeffs)

        from srxraylib.plot.gol import plot
        plot(ji_array, kv(nu, ji_array),
             ji_array, kv_approx_coeff_fine(res.x, ji_array, nu=nu),
             legend=['k%f'%nu,'k approx'], xlog=0, ylog=0, yrange=[1e-18,10])


    if True:

        # display comparison

        ji_interval_number_of_points = 101
        ji_array = numpy.linspace(01e-10, 10, ji_interval_number_of_points)


        from srxraylib.plot.gol import plot
        for nu in [1/3, 2/3, 5/3]:
            plot(ji_array, kv(nu, ji_array),
                 ji_array, kv_approx_fine(nu, ji_array),
                 legend=['k%f'%nu,'k approx'], xlog=0, ylog=0, yrange=[1e-18,10])