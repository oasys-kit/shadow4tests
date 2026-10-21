__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Polarization of the rays of the (planar) undulator source.
#
# The emission codes (internal, pySRU, SRW) calculate the field in a frame x (horizontal), y (vertical), z (beam),
# right-handed, which is the SHADOW standard frame (u_sigma, -u_pi, v) rotated by 180 deg around the beam: the SHADOW
# direction (X', Z') corresponds to (x, y) = (-X', Z'). The emission is calculated in one quadrant and the polarization
# of the rays in the other quadrants is obtained by symmetry: P2 is odd in x and y, P3 is odd in y and even in x (the
# mirror symmetry in x includes a time reversal). The radiation is mainly linearly polarized, with P2 changing sign
# with the quadrant; a small P3 appears off axis (finite number of periods, photon energy off resonance).
#
# The reference is the time-domain Lienard-Wiechert far field of the electron trajectory (Fourier transform with
# exp(+i w t), so that the field is Re[E(w) exp(-i w t)], as in SHADOW).
#

import unittest
import numpy
import scipy.constants as codata

from shadow4.sources.s4_electron_beam import S4ElectronBeam
from shadow4.sources.undulator.s4_undulator import S4Undulator
from shadow4.sources.undulator.s4_undulator_light_source import S4UndulatorLightSource
from shadow4.sources.undulator.calculate_undulator_emission import \
    _pysru_analytical_trajectory_plane_undulator, _pysru_energy_radiated_approximation_and_farfield, _stokes_from_fields

E_GEV, K, LAMBDA_U, N_PERIODS = 6.0, 1.68, 0.018, 111
GAMMA = E_GEV * 1e9 / 0.511e6
BETA_ET = numpy.sqrt(1 - 1 / GAMMA ** 2) * (1 - (K / (2 * GAMMA)) ** 2)
PHOTON_ENERGY = 0.995 * 2 * GAMMA ** 2 * codata.h * codata.c / codata.e / LAMBDA_U / (1 + K ** 2 / 2)
OMEGA = 2 * numpy.pi * PHOTON_ENERGY * codata.e / codata.h


def trajectory(points_per_period=100, n_periods=N_PERIODS):
    return _pysru_analytical_trajectory_plane_undulator(K=K, gamma=GAMMA, lambda_u=LAMBDA_U, Nb_period=n_periods,
                                                        Nb_point=points_per_period, Beta_et=BETA_ET)


def stokes_field(T, x, y, omega=OMEGA, D=100.0):
    """P1, P2, P3 of the field of the emission code at the angles (x, y) (frame of the emission code)."""
    E = _pysru_energy_radiated_approximation_and_farfield(omega=omega, trajectory=T, x=x * D, y=y * D, D=D)
    return _stokes_from_fields(E[0], E[1])


def stokes_lienard_wiechert(T, x, y, omega=OMEGA):
    t, X, Y, Z, vx, vy, vz, ax, ay, az = T
    r = numpy.vstack((X, Y, Z)) * codata.c
    b = numpy.vstack((vx, vy, vz))
    bd = numpy.vstack((ax, ay, az))
    n = numpy.array([x, y, 1.0])
    n /= numpy.linalg.norm(n)
    E = numpy.cross(n[None, :], numpy.cross((n[:, None] - b).T, bd.T)) / ((1 - n @ b) ** 3)[:, None]
    tobs = t - (n @ r) / codata.c
    integrand = E[:, :2] * numpy.exp(1j * omega * tobs)[:, None]
    J = ((integrand[1:] + integrand[:-1]) * 0.5 * numpy.diff(tobs)[:, None]).sum(axis=0)  # trapezoidal rule
    return _stokes_from_fields(J[0], J[1])


def undulator_beam(code='internal', nrays=5000):
    electron_beam = S4ElectronBeam(energy_in_GeV=E_GEV, energy_spread=0.001, current=0.2)
    electron_beam.set_sigmas_all(sigma_x=0, sigma_y=0, sigma_xp=0, sigma_yp=0)
    undulator = S4Undulator(K_vertical=K, period_length=LAMBDA_U, number_of_periods=N_PERIODS,
                            emin=PHOTON_ENERGY, emax=PHOTON_ENERGY, ng_e=1, maxangle=2 / GAMMA,
                            ng_t=51, ng_p=21, ng_j=50, code_undul_phot=code, flag_emittance=0, flag_size=0)
    light_source = S4UndulatorLightSource(electron_beam=electron_beam, magnetic_structure=undulator,
                                          nrays=nrays, seed=5676561)
    return light_source.get_beam()


ANGLES = [(2.3e-4, 1.0e-4), (2.5e-4, 5.2e-5), (9.5e-5, 9.2e-5), (1.5e-4, 1.0e-4)]


class TestUndulatorPolarization(unittest.TestCase):

    def test_field_symmetry(self):
        # E_x even and E_y odd in y (trajectory in the plane y = 0): exact, also where the field is weak
        T = trajectory(points_per_period=20)
        for x, y in ANGLES:
            P_up, P_down = stokes_field(T, x, y), stokes_field(T, x, -y)
            numpy.testing.assert_allclose(P_up * [1, -1, -1], P_down, atol=1e-10)
        # in x: P2 odd, P1 and P3 even (approximately, finite number of periods)
        T = trajectory(points_per_period=50)
        for x, y in [(4e-5, 3e-5), (9.5e-5, 9.2e-5)]:
            numpy.testing.assert_allclose(stokes_field(T, x, y) * [1, -1, 1], stokes_field(T, -x, y), atol=2e-3)

    def test_field_against_lienard_wiechert(self):
        T = trajectory(points_per_period=200)
        for x, y in ANGLES:
            numpy.testing.assert_allclose(stokes_field(T, x, y), stokes_lienard_wiechert(T, x, y), atol=0.02,
                                          err_msg="angles %g %g" % (x, y))

    def test_rays(self):
        beam = undulator_beam()
        xp, zp = beam.get_column(4, nolost=1), beam.get_column(6, nolost=1)
        P = numpy.array([beam.get_column(col, nolost=1) for col in [42, 43, 44]])
        assert numpy.allclose((P ** 2).sum(axis=0), 1.0)                     # fully polarized rays
        # quadrant symmetry: P2 > 0 for X' Z' > 0 (Lienard-Wiechert in the SHADOW frame), and the beam is symmetric
        self.assertGreater(P[1][xp * zp > 0].mean(), 0.05)
        self.assertLess(P[1][xp * zp < 0].mean(), -0.05)
        self.assertLess(abs(P[1].mean()), 0.01)
        self.assertLess(abs(P[2].mean()), 0.01)
        # mainly sigma polarization in the central cone
        central = numpy.sqrt(xp ** 2 + zp ** 2) < 0.5 / GAMMA
        self.assertGreater(P[0][central].mean(), 0.95)
        # P3 is odd in Z' and even in X'
        self.assertLess(abs(P[2][(xp > 0) & (zp > 0)].mean() - P[2][(xp < 0) & (zp > 0)].mean()), 0.02)
        self.assertLess(abs(P[2][zp > 0].mean() + P[2][zp < 0].mean()), 0.02)
        # rays against the field at their direction: (x, y) = (-X', Z')
        T = trajectory(points_per_period=50)
        rng = numpy.random.default_rng(0)
        err = [numpy.abs(stokes_field(T, -xp[i], zp[i]) - P[:, i]).max() for i in rng.choice(xp.size, 100, replace=False)]
        self.assertLess(numpy.median(err), 0.01)
        self.assertLess(numpy.percentile(err, 75), 0.03)


if __name__ == "__main__":
    unittest.main()
