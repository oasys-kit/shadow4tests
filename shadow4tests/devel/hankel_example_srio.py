"""
Typical usage
=================

To demonstrate the use of the Hankel Transform class, we will give an example
of propagating a radially-symmetric beam using the beam propagation method.

In this case, it will be a simple Gaussian beam propagating way from focus and
diverging.

First we will use a loop over :math:`z` position, and then we will demonstrate
the vectorisation of the :func:`.HankelTransforms.iqdht` (and
:func:`~.HankelTransforms.qdht`) functions.

"""

from srxraylib.plot.gol import set_qt, plot, plot_image
set_qt()

# %%
# First import the standard libraries
import matplotlib.pyplot as plt
import numpy as np

# %%
# Then the functions from this package
from pyhank import HankelTransform
# noinspection PyUnresolvedReferences
# from helper import gauss1d, imagesc


"""
Helper functions
----------------

Defines a couple of helper functions for the examples.
"""

import numpy as np
import matplotlib.pyplot as plt


# 1D Gaussian function
def gauss1d(x, x0, fwhm):
    return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)

def slit1d(x, x0, fwhm):
    y = np.ones_like(x)
    y[ (np.abs(x - x0) > fwhm)] = 0
    return y


# Plotting function equivalent to Matlab's imagesc
def imagesc(x: np.ndarray, y: np.ndarray, intensity: np.ndarray, axes=None, **kwargs):
    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1d arrays"
    assert intensity.ndim == 2, "Intensity must be a 2d array"
    extent = (x[0], x[-1], y[-1], y[0])
    if axes is None:
        img = plt.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    else:
        img = axes.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    img.axes.invert_yaxis()
    return img

if __name__ == "__main__":
    # %%
    # Initialise radius grid
    nr = 1024 * 5  # Number of sample points
    r_max = 5e-3 * 5 # Maximum radius (5mm)
    r = np.linspace(0, r_max, nr)

    # %%
    # Initialise :math:`z` grid
    Nz = 20  # Number of z positions
    z_max = 100.0  # Maximum propagation distance
    z = np.linspace(0, z_max, Nz)  # Propagation axis

    # %%
    # Set up beam parameters
    Dr = 100e-6  / 2 # Beam radius (100um)
    lambda_ = 1.5e-10 # 488e-9  # wavelength 488nm
    k0 = 2 * np.pi / lambda_  # Vacuum k vector

    # %%
    # Set up a :class:`.HankelTransform` object, telling it the order (``0``) and
    # the radial grid.
    H = HankelTransform(order=0, radial_grid=r)

    # %%
    # Set up the electric field profile at :math:`z = 0`, and resample onto the correct radial grid
    # (``transformer.r``) as required for the QDHT.
    # Er = gauss1d(r, 0, Dr)  + 0j  # Initial field
    Er = slit1d(r, 0, Dr)  + 0j  # Initial field


    ErH = H.to_transform_r(Er)  # Resampled field

    # # Now plot an image showing the intensity as a function of
    # # radius and propagation distance.
    if 0:
        plot(r * 1e3, np.abs(Er) ** 2,
             r * 1e3, np.unwrap(np.angle(Er)),
             H.r * 1e3, np.abs(ErH) ** 2,
             H.r * 1e3, np.unwrap(np.angle(ErH)),
             xrange=[0,1], yrange=[0,1],
             xtitle="r [mm]", ytitle="Field intensity /arb",title="Initial electric field distribution",
             legend=['$|E(r)|^2$', '$\\phi(r)$', '$|E(H.r)|^2$', '$\\phi(H.r)$'],
             marker=[None,None,None,'+'], linestyle=[None,None,None,''],
             )

    # %%
    # Perform Hankel Transform
    # ------------------------

    # Convert from physical field to physical wavevector
    EkrH = H.qdht(ErH)

    if 0:
        plot(H.kr, np.abs(EkrH) ** 2,
             # xrange=[0,1], yrange=[0,1],
             xtitle=r'Radial wave-vector ($k_r$) /rad $m^{-1}$', ytitle='Field intensity /arb.',title="Radial wave-vector distribution",
             )


    # %%
    # Propagate the beam - loop
    # -------------------------
    # Do the propagation in a loop over :math:`z`

    # Pre-allocate an array for field as a function of r and z
    Erz = np.zeros((nr, Nz), dtype=complex)
    kz = np.sqrt(k0 ** 2 - H.kr ** 2)
    print(">>>>>000 kz", kz.shape)

    for i, z_loop in enumerate(z):
        phi_z = kz * z_loop  # Propagation phase
        EkrHz = EkrH * np.exp(1j * phi_z)  # Apply propagation
        print("   >>>>", i, EkrHz.shape)
        ErHz = H.iqdht(EkrHz)  # iQDHT
        Erz[:, i] = H.to_original_r(ErHz)  # Interpolate output
    Irz = np.abs(Erz) ** 2


    # # %%
    # # Now plot an image showing the intensity as a function of
    # # radius and propagation distance.
    #


    # plt.figure()
    # imagesc(z * 1e3, r * 1e3, Irz)
    # plt.title('Radial field intensity as a function of propagation for annular beam')
    # plt.xlabel('Propagation distance ($z$) /mm')
    # plt.ylabel('Radial position ($r$) /mm')
    # plt.ylim([0, 1])
    # plt.show()

    if 0:
        plot_image(Irz.T, z, r * 1e3,
                   title='***Radial field intensity as a function of propagation for annular beam',
                   xtitle=r'Propagation distance ($z$) /m',
                   ytitle=r'Radial position ($r$) /mm', aspect='auto',
                   yrange=[0,1])


    # # %%
    # # The plot above shows a reduction of intensity with :math:`z`, but it is
    # # bit difficult to see the beam growing in :math:`r`. To show that, let's
    # # plot the intensity normalised such that the peak intensity at each :math:`z`
    # # coordinate is the same.
    Irz_norm = Irz / Irz[0, :]
    #
    # plt.figure()
    # imagesc(z * 1e3, r * 1e3, Irz_norm)
    # plt.xlabel('Propagation distance ($z$) /mm')
    # plt.ylabel('Radial position ($r$) /mm')
    # plt.ylim([0, 1])
    # plt.show()

    if 0:
        plot_image(Irz_norm.T, z, r * 1e3,
                   title='***Radial field intensity as a function of propagation for annular beam',
                   xtitle=r'Propagation distance ($z$) /m',
                   ytitle=r'Radial position ($r$) /mm', aspect='auto',
                   yrange=[0,1])



    #
    # # %%
    # # Propagate the beam - vectorised
    # # -------------------------------
    kz = np.sqrt(k0 ** 2 - H.kr ** 2)
    print(">>>>>*** kz", kz.shape)
    phi_z = kz[:, np.newaxis] * z[np.newaxis, :]  # Propagation phase
    EkrHz = EkrH[:, np.newaxis] * np.exp(1j * phi_z)  # Apply propagation
    print("   >>>>***", EkrHz.shape)
    ErHz = H.iqdht(EkrHz)  # iQDHT
    Erz = H.to_original_r(ErHz)  # Interpolate output
    Irz_vectorised = np.abs(Erz) ** 2

    #
    # # %%
    # # Now plot the result to check it is the same as the loop approach
    # plt.figure()
    # imagesc(z * 1e3, r * 1e3, Irz_vectorised)
    # plt.title('Radial field intensity as a function of propagation for annular beam')
    # plt.xlabel('Propagation distance ($z$) /mm')
    # plt.ylabel('Radial position ($r$) /mm')
    # plt.ylim([0, 1])
    # plt.show()
    # # %%
    # # Assert the two approaches produce the same intensity
    assert np.allclose(Irz, Irz_vectorised)

    if 0:
        plot_image(Irz_vectorised.T, z, r * 1e3,
                   title='***VECT** Radial field intensity as a function of propagation for annular beam',
                   xtitle=r'Propagation distance ($z$) /m',
                   ytitle=r'Radial position ($r$) /mm', aspect='auto',
                   yrange=[0, 0.2], show=0)

        # plot(H.r * 1e3, np.abs(ErH) ** 2, show=1, title="test")

    #theory
    from scipy.special import jv
    sin_theta_array = np.sin(r / z_max)
    aperture_diameter = 2 * Dr
    x = (2 * np.pi / lambda_) * (aperture_diameter / 2) * sin_theta_array
    x_over_pi = x / np.pi
    electric_field = 2 * jv(1, x) / x
    intensity = electric_field ** 2

    plot(r * 1e3, np.abs(Er)**2,
        H.r * 1e3, np.abs(ErH) ** 2,
         r * 1e3, Irz_vectorised.T[0,:],
         r * 1e3, Irz_vectorised.T[-1, :] / (Irz_vectorised.T[-1, :]).max(),
         r * 1e3, intensity,
         xrange=[0.,0.5], legend=['data','initial', '0', '-1 renormalized','theory'], show=0)

    plot(np.degrees(r / z_max), np.abs(Er)**2,
         np.degrees(r / z_max), np.abs(ErH) ** 2,
         np.degrees(r / z_max), Irz_vectorised.T[0,:],
         np.degrees(r / z_max), Irz_vectorised.T[-1, :] / (Irz_vectorised.T[-1, :]).max(),
         np.degrees(r / z_max), intensity,
         xrange=[0.,0.00015], legend=['data','initial', '0', '-1 renormalized','theory'])
