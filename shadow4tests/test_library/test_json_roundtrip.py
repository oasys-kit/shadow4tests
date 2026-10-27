__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"

#
# Json files of shadow4 beamlines (S4Beamline.to_json() and shadow4.tools.json_tools.load_from_json_*).
#
# The SYNED mechanism alone (default constructor + values of the keys) did not restore the quantities derived in the
# constructors of the shadow4 optical elements: e.g., a spherical mirror was read back with the optical surface of the
# default constructor (wrong ray tracing), and to_python_code() used the default inputs. Moreover, the movements of
# the beamline elements were not written, and some keys of SourceGeometrical were wrong ("f_color ", "f_foher " with
# a trailing space; the units of the photon energy, f_phot, were not written).
#
# Now the arguments of the constructors of the optical elements and the movements are written in the json file, and
# the optical elements are rebuilt with their constructors. Checked here:
#   - a beamline (geometrical source, plane and spherical mirrors, with movements) is read back with the same ray
#     tracing, the same python code, the movements and the source flags, and is written again identically;
#   - every shadow4 optical element class is read back with the same python code and the same optical surface;
#   - special elements (CRL, lens, additional numerical mesh, compound, DABAX objects);
#   - files written by previous versions (without the constructor arguments): the optical surface of curved
#     elements is restored from the SYNED surface shape.
#

import unittest
import importlib, inspect, pkgutil, contextlib, io, json, warnings
import json_tricks
import numpy

import shadow4
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import Rectangle, Ellipse
from shadow4.tools.json_tools import load_from_json_text
from shadow4.beamline.s4_beamline import S4Beamline
from shadow4.beamline.s4_beamline_element import S4BeamlineElement
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.beamline.s4_optical_element_decorators import S4OpticalElementDecorator
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical
from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import S4PlaneMirror, S4PlaneMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_sphere_mirror import S4SphereMirror, S4SphereMirrorElement
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror
from shadow4.beamline.optical_elements.refractors.s4_crl import S4CRL
from shadow4.beamline.optical_elements.refractors.s4_lens import S4Lens
from shadow4.beamline.optical_elements.compound.s4_compound import S4Compound
from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import S4PlaneCrystal


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def get_beamline():
    src = SourceGeometrical(name="my source", nrays=2000, seed=11)
    src.set_spatial_type_gaussian(sigma_h=1e-5, sigma_v=2e-5)
    src.set_angular_distribution_gaussian(sigdix=1e-4, sigdiz=2e-4)
    src.set_energy_distribution_singleline(1.5498, unit='A')
    src.set_polarization(polarization_degree=0.7, phase_diff=0.3, coherent_beam=1)
    bl = S4Beamline(light_source=src)
    m1 = S4PlaneMirror(name="M1", boundary_shape=Rectangle(-0.01, 0.02, -0.1, 0.15),
                       f_reflec=1, f_refl=5, coating_material="Rh", coating_density=12.4)
    bl.append_beamline_element(S4PlaneMirrorElement(m1, ElementCoordinates(
        p=10, q=2, angle_radial=numpy.radians(89.8), angle_azimuthal=0, angle_radial_out=numpy.radians(89.8))))
    m2 = S4SphereMirror(name="M2", surface_calculation=0, is_cylinder=1, cylinder_direction=0, convexity=1,
                        p_focus=12, q_focus=5, grazing_angle=3e-3)
    bl.append_beamline_element(S4SphereMirrorElement(m2, ElementCoordinates(
        p=2, q=5, angle_radial=numpy.pi / 2 - 3e-3, angle_azimuthal=numpy.pi / 2, angle_radial_out=numpy.pi / 2 - 3e-3),
        movements=S4BeamlineElementMovements(f_move=1, offset_x=1e-5, rotation_x=1e-5)))
    m3 = S4SphereMirror(name="M3", boundary_shape=Ellipse(-0.01, 0.01, -0.05, 0.05), surface_calculation=1,
                        is_cylinder=0, convexity=1, radius=500.0, f_reflec=1, f_refl=1, refraction_index=1 - 1e-5 + 1e-7j)
    bl.append_beamline_element(S4SphereMirrorElement(m3, ElementCoordinates(
        p=1, q=3, angle_radial=numpy.radians(89.7), angle_azimuthal=numpy.radians(30.0),
        angle_radial_out=numpy.radians(89.7))))
    return bl


def trace(beamline):
    beam = beamline.get_light_source().get_beam()
    for i in range(beamline.get_beamline_elements_number()):
        element = beamline.get_beamline_element_at(i)
        element.set_input_beam(beam)
        beam, _ = element.trace_beam()
    return beam


def conic_coefficients(optical_element):
    try:
        with quiet(): surface = optical_element.get_optical_surface_instance()
        return numpy.array(surface.ccc) if hasattr(surface, "ccc") else None
    except Exception:
        return None


def roundtrip_element(optical_element):
    element = S4BeamlineElement(optical_element=optical_element, coordinates=ElementCoordinates(p=1, q=2))
    with quiet(): text = element.to_json()
    with quiet(): return text, load_from_json_text(text).get_optical_element()


def optical_element_classes():
    warnings.filterwarnings("ignore")
    classes = []
    for _, module_name, _ in pkgutil.walk_packages(shadow4.__path__, "shadow4."):
        if "optical_elements" not in module_name: continue
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module_name and issubclass(cls, S4OpticalElementDecorator):
                classes.append(cls)
    return classes


class TestJsonBeamline(unittest.TestCase):

    def test_beamline_roundtrip(self):
        bl = get_beamline()
        with quiet(): text = bl.to_json()
        with quiet(): bl2 = load_from_json_text(text)
        b0, b1 = trace(bl), trace(bl2)
        assert (b0.rays[:, 9] > 0).sum() > 100
        numpy.testing.assert_array_equal(b0.rays, b1.rays)
        self.assertEqual(bl.to_python_code(), bl2.to_python_code())
        # movements
        movements = bl2.get_beamline_element_at(1).get_movements()
        self.assertEqual((movements.f_move, movements.offset_x, movements.rotation_x), (1, 1e-5, 1e-5))
        self.assertIsNone(bl2.get_beamline_element_at(0).get_movements())
        # the sphere defined by p, q, grazing angle is kept as such
        self.assertEqual(bl2.get_beamline_element_at(1).get_optical_element()._s4_init_kwargs["p_focus"], 12)
        # source flags
        source = bl2.get_light_source()
        self.assertEqual((source._f_foher, source._f_phot, source._f_color), (1, 1, 1))
        d = json.loads(text)["light_source"]
        for key in ("f_color", "f_foher", "f_phot"):
            self.assertIn(key, d)
        self.assertNotIn("f_color ", d)
        # written again identically
        with quiet(): self.assertEqual(json.loads(bl2.to_json()), json.loads(text))


class TestJsonOpticalElements(unittest.TestCase):

    def test_all_optical_element_classes(self):
        n_checked = 0
        for cls in optical_element_classes():
            try:
                with quiet():
                    oe = cls()
                    code = oe.to_python_code()
            except Exception:
                continue   # default instance not usable (e.g., abstract classes, or elements needing arguments)
            text, oe2 = roundtrip_element(oe)
            self.assertIs(type(oe2), cls)
            with quiet(): self.assertEqual(code, oe2.to_python_code(), msg=cls.__name__)
            c0, c1 = conic_coefficients(oe), conic_coefficients(oe2)
            if c0 is not None:
                numpy.testing.assert_allclose(c1, c0, equal_nan=True, err_msg=cls.__name__)
            n_checked += 1
        self.assertGreater(n_checked, 45)

    def test_special_elements(self):
        x = numpy.linspace(-0.01, 0.01, 11)
        y = numpy.linspace(-0.1, 0.1, 21)
        cases = [
            S4CRL(name="crl", n_lens=5, piling_thickness=2.5e-3, surface_shape=2, convex_to_the_beam=0,
                  ri_calculation_mode=0, prerefl_file="", refraction_index=1 - 3.4e-6, attenuation_coefficient=10.0,
                  thickness=50e-6, radius=200e-6),
            S4Lens(name="lens", surface_shape=1, convex_to_the_beam=1, ri_calculation_mode=0, prerefl_file="",
                   refraction_index=1.5, thickness=1e-3, radius=0.05),
            S4AdditionalNumericalMeshMirror(name="mesh",
                ideal_mirror=S4SphereMirror(name="ideal", surface_calculation=1, radius=300.0),
                numerical_mesh_mirror=S4NumericalMeshMirror(name="m", xx=x, yy=y,
                                                            zz=1e-9 * numpy.outer(numpy.sin(50 * x), numpy.cos(20 * y)))),
            S4Compound(name="compound", oe_list=[
                S4PlaneMirrorElement(S4PlaneMirror(name="c1", boundary_shape=Rectangle(-0.01, 0.01, -0.1, 0.1)),
                                     ElementCoordinates(p=1, q=0.5, angle_radial=numpy.radians(89.8))),
                S4SphereMirrorElement(S4SphereMirror(name="c2", surface_calculation=1, radius=600.0),
                                      ElementCoordinates(p=0.5, q=1, angle_radial=numpy.radians(89.8)),
                                      movements=S4BeamlineElementMovements(f_move=1, rotation_y=1e-4))]),
        ]
        try:
            from dabax.dabax_xraylib import DabaxXraylib
            cases.append(S4PlaneCrystal(name="xtal", material="Si", miller_index_h=3, miller_index_k=3, miller_index_l=3,
                                        f_central=1, phot_cent=10000.0, material_constants_library_flag=1,
                                        dabax=DabaxXraylib(file_f1f2="f1f2_Henke.dat")))
            cases.append(S4PlaneMirror(name="m", f_reflec=1, f_refl=6, coating_material="Pt", coating_density=21.4,
                                       dabax=DabaxXraylib(file_f1f2="f1f2_Sasaki.dat")))
        except ImportError:
            pass
        for oe in cases:
            text, oe2 = roundtrip_element(oe)
            self.assertIs(type(oe2), type(oe))
            with quiet(): self.assertEqual(oe.to_python_code(), oe2.to_python_code(), msg=type(oe).__name__)
            if getattr(oe, "_dabax", None) is not None:
                self.assertEqual(oe2._dabax.get_file_f1f2(), oe._dabax.get_file_f1f2())
            if isinstance(oe, S4Compound):
                movements = oe2._oe_list[1].get_movements()
                self.assertEqual((movements.f_move, movements.rotation_y), (1, 1e-4))

    def test_files_of_previous_versions(self):
        # without the constructor arguments (files written by previous versions), the optical surface of curved
        # elements is set from the SYNED surface shape stored in the file (known limitation: the SYNED paraboloid
        # does not contain all the information used by shadow4, so paraboloids are not included)
        from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
        from shadow4.beamline.optical_elements.mirrors.s4_hyperboloid_mirror import S4HyperboloidMirror
        from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import S4ToroidMirror
        cases = [
            S4SphereMirror(name="s1", surface_calculation=1, is_cylinder=0, convexity=1, radius=500.0),
            S4SphereMirror(name="s2", surface_calculation=0, is_cylinder=1, p_focus=12, q_focus=5, grazing_angle=3e-3),
            S4EllipsoidMirror(name="e", surface_calculation=0, is_cylinder=1, p_focus=20, q_focus=3, grazing_angle=3e-3),
            S4HyperboloidMirror(name="h", surface_calculation=0, is_cylinder=1, p_focus=20, q_focus=3,
                                grazing_angle=3e-3),
            S4ToroidMirror(name="t", surface_calculation=1, min_radius=0.1, maj_radius=300.0),
        ]
        for oe in cases:
            c0 = conic_coefficients(oe)
            text, _ = roundtrip_element(oe)
            d = json_tricks.loads(text)
            d["optical_element"].pop("s4_init_kwargs")
            with quiet(): oe3 = load_from_json_text(json_tricks.dumps(d)).get_optical_element()
            if c0 is not None:
                numpy.testing.assert_allclose(conic_coefficients(oe3), c0, err_msg=oe.get_name())
            else: # toroid: no conic coefficients, compare the radii of the surface
                with quiet():
                    s0, s3 = oe.get_optical_surface_instance(), oe3.get_optical_surface_instance()
                numpy.testing.assert_allclose([s3.r_min, s3.r_maj], [s0.r_min, s0.r_maj])


if __name__ == "__main__":
    unittest.main()
