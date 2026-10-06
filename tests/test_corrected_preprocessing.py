"""Numerical regression tests for preprocessing, without detector inference."""

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/02/preprocessing.py"
SPEC = importlib.util.spec_from_file_location("corrected_preprocessing", MODULE_PATH)
preprocessing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocessing)


class CorrectedPreprocessingTests(unittest.TestCase):
    def test_low_light_darkens_midtones_and_preserves_endpoints(self):
        image = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        dark = preprocessing.degrade_image(image, "low_light", np.random.default_rng(42))
        self.assertTrue(np.all(dark[:, 1:-1] < image[:, 1:-1]))
        np.testing.assert_array_equal(dark[:, [0, -1]], image[:, [0, -1]])

    def test_degradations_use_supplied_rng_reproducibly(self):
        image = np.full((8, 9, 3), 100, dtype=np.uint8)
        for condition in preprocessing.CONDITIONS:
            with self.subTest(condition=condition):
                first = preprocessing.degrade_image(image, condition, np.random.default_rng(19))
                np.random.seed(9876)
                second = preprocessing.degrade_image(image, condition, np.random.default_rng(19))
                np.testing.assert_array_equal(first, second)
        first = preprocessing.degrade_image(image, "gaussian_noise", np.random.default_rng(19))
        second = preprocessing.degrade_image(image, "gaussian_noise", np.random.default_rng(20))
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(image, np.full_like(image, 100))

    def test_multifactor_uses_corrected_gamma(self):
        image = np.full((5, 6, 3), 110, dtype=np.uint8)
        expected = preprocessing.add_haze(preprocessing.apply_gamma(
            preprocessing.add_gaussian_noise(image, np.random.default_rng(5)), gamma=2.0))
        actual = preprocessing.degrade_image(image, "multifactor", np.random.default_rng(5))
        np.testing.assert_array_equal(actual, expected)

    def test_degradation_rejects_unknown_condition_and_implicit_rng(self):
        image = np.full((5, 6, 3), 110, dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocessing.degrade_image(image, "unknown", np.random.default_rng(5))
        with self.assertRaises(ValueError):
            preprocessing.degrade_image(image, "gaussian_noise", None)

    def test_diffusion_has_no_opposite_edge_coupling(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        image[:, 0] = 20
        result = preprocessing.anisotropic_diffusion(image, niter=1)
        np.testing.assert_array_equal(result[:, -1], 0)
        self.assertTrue(np.all(result[:, 1] > 0))
        self.assertTrue(np.all(result[:, 0] < 20))
        rotated = preprocessing.anisotropic_diffusion(np.swapaxes(image, 0, 1), niter=1)
        np.testing.assert_array_equal(rotated[-1], 0)

    def test_diffusion_preserves_constant_fields_and_zero_iterations(self):
        for shape in ((8, 8, 3), (1, 8, 3), (8, 1, 3), (1, 1, 3)):
            image = np.full(shape, 137, dtype=np.uint8)
            np.testing.assert_array_equal(preprocessing.anisotropic_diffusion(image), image)
        image = np.arange(60, dtype=np.uint8).reshape((4, 5, 3))
        np.testing.assert_array_equal(preprocessing.anisotropic_diffusion(image, niter=0), image)

    def test_diffusion_rejects_invalid_parameters(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        cases = ({"kappa": 0}, {"kappa": np.inf}, {"niter": -1}, {"niter": 1.5},
                 {"niter": True}, {"gamma": 0}, {"gamma": 0.251}, {"gamma": np.nan})
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                preprocessing.anisotropic_diffusion(image, **arguments)

    def test_constant_log_and_correlation_are_safe(self):
        image = np.full((9, 9, 3), 137, dtype=np.uint8)
        log_map = preprocessing.compute_log_map(image)
        np.testing.assert_array_equal(log_map, np.zeros((9, 9)))
        correlation = preprocessing.compute_correlations(image, log_map)
        self.assertIsNone(correlation["rho_log_g"])
        self.assertEqual(correlation["correlation_status"], "constant_log_map")
        self.assertIn("null", json.dumps(correlation, allow_nan=False))

    def test_constant_gradient_and_nonfinite_map_are_safe(self):
        image = np.full((3, 3, 3), 50, dtype=np.uint8)
        log_map = np.arange(9, dtype=float).reshape((3, 3)) / 8
        correlation = preprocessing.compute_correlations(image, log_map)
        self.assertIsNone(correlation["rho_log_g"])
        self.assertEqual(correlation["correlation_status"], "constant_gradient_map")
        log_map[0, 0] = np.nan
        correlation = preprocessing.compute_correlations(image, log_map)
        self.assertIsNone(correlation["rho_log_g"])
        self.assertEqual(correlation["correlation_status"], "non_finite_map")

    def test_nondegenerate_correlation_is_finite(self):
        image = np.random.default_rng(4).integers(0, 256, (11, 12, 3), dtype=np.uint8)
        result = preprocessing.compute_correlations(image, preprocessing.compute_log_map(image))
        self.assertEqual(result["correlation_status"], "ok")
        self.assertTrue(-1 <= result["rho_log_g"] <= 1)

    def test_brightness_matches_quantized_clipped_target(self):
        image = np.arange(256, dtype=np.uint8).reshape((16, 16, 1)).repeat(3, axis=2)
        log_map = np.random.default_rng(123).uniform(0, 1, (16, 16))
        target = preprocessing.apply_log_modulation(image, log_map)
        matched, diagnostics = preprocessing.brightness_matched_control(image, target)
        self.assertLessEqual(abs(float(matched.mean()) - float(target.mean())), 0.5)
        self.assertAlmostEqual(diagnostics["achieved_mean_intensity"], float(matched.mean()))
        self.assertAlmostEqual(diagnostics["target_mean_intensity"], float(target.mean()))
        self.assertAlmostEqual(diagnostics["mean_intensity_error"], float(matched.mean() - target.mean()))
        expected = np.clip(image.astype(float) * diagnostics["gain"], 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(matched, expected)
        self.assertTrue(np.any(matched == 255))

    def test_brightness_reports_unavoidable_quantization_residual(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        target = image.copy()
        target[0] += 1
        matched, diagnostics = preprocessing.brightness_matched_control(image, target)
        self.assertEqual(diagnostics["absolute_mean_intensity_error"], 0.5)
        self.assertEqual(diagnostics["status"], "nearest_quantized_mean")
        self.assertEqual(np.unique(matched).size, 1)

    def test_brightness_handles_black_and_fully_saturated_images(self):
        for level in (0, 255):
            image = np.full((4, 4, 3), level, dtype=np.uint8)
            matched, diagnostics = preprocessing.brightness_matched_control(image, image)
            np.testing.assert_array_equal(matched, image)
            self.assertEqual(diagnostics["gain"], 1)
            self.assertEqual(diagnostics["absolute_mean_intensity_error"], 0)

    def test_brightness_rejects_unattainable_targets(self):
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        for target in (np.full_like(image, 99), np.full_like(image, 131)):
            with self.assertRaises(ValueError):
                preprocessing.brightness_matched_control(image, target)

    def test_prepare_methods_share_one_ad_and_log_with_serializable_diagnostics(self):
        image = np.random.default_rng(99).integers(0, 256, (13, 14, 3), dtype=np.uint8)
        original = image.copy()
        with patch.object(preprocessing, "anisotropic_diffusion", wraps=preprocessing.anisotropic_diffusion) as ad:
            with patch.object(preprocessing, "compute_log_map", wraps=preprocessing.compute_log_map) as log:
                methods, diagnostics = preprocessing.prepare_methods(image)
        ad.assert_called_once()
        log.assert_called_once()
        self.assertEqual(tuple(methods), preprocessing.METHODS)
        for name, output in methods.items():
            with self.subTest(method=name):
                self.assertEqual(output.dtype, np.uint8)
                self.assertEqual(output.shape, image.shape)
                stats = diagnostics["method_intensity"][name]
                self.assertEqual(stats["mean_intensity"], float(output.mean()))
                self.assertEqual(stats["saturated_fraction"], float(np.mean(output == 255)))
        np.testing.assert_array_equal(methods["baseline"], original)
        np.testing.assert_array_equal(image, original)
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
