#
# Tools for the tests of curved mirrors (test_mirrors_branch5_*.py): a geometrical source and the tracing of a mirror
# in the usual SHADOW4 way (element with coordinates p, q and grazing angle).
#
import numpy
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from syned.beamline.element_coordinates import ElementCoordinates


def source_beam(sigma=0.0, divergence=1e-4, nrays=5000, seed=5676561):
    """Geometrical source: point (sigma=0) or Gaussian size sigma, Gaussian divergence (rad), 1 keV."""
    source = SourceGeometrical(nrays=nrays, seed=seed)
    if sigma == 0: source.set_spatial_type_point()
    else: source.set_spatial_type_gaussian(sigma_h=sigma, sigma_v=sigma)
    source.set_angular_distribution_gaussian(sigdix=divergence, sigdiz=divergence)
    source.set_energy_distribution_singleline(1000.0)
    return source.get_beam()


def trace_mirror(element_class, optical_element, p, q, grazing_angle, beam):
    """Traces the beam through the mirror; returns the beam at the image plane at distance q."""
    element = element_class(optical_element=optical_element,
                            coordinates=ElementCoordinates(p=p, q=q, angle_radial=numpy.pi / 2 - grazing_angle),
                            input_beam=beam.duplicate())
    beam_out, _ = element.trace_beam()
    return beam_out


def columns(beam, cols=(1, 3, 4, 5, 6)):
    """Columns of the good rays."""
    return [beam.get_column(c, nolost=1) for c in cols]
