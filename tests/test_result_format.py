"""Result-presentation checks using the standard library and recorded JSON only."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("result_formatter", ROOT / "format_results.py")
formatter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formatter)
FULL_RUN = ROOT / "experiments/02/runs/full_20261006T031903Z_33e4b749"
ARCHIVE_HASHES = {
    "pilot_5": "8265a90458cca09c989245e2ec0610aa1f4e674e8e0bef37eb900cf0759fbe8e",
    "full_5000": "287c188061661b1d7702895b434fc18aa095496fb8ea5c8382a5c981c1802bb4",
}


def legacy_dir(root, name):
    return root / "experiments/01/runs" / name


def legacy_result(name="pilot_5"):
    return formatter.read_json(legacy_dir(ROOT, name) / "results.json")


def complete_fixture():
    data = legacy_result()
    data.update(run_kind="full", image_count=2, inference_calls=48,
                artifact_sha256={"ignored/raw.jsonl.gz": "not needed for presentation"})
    for methods in data["map_results"].values():
        methods["ad_brightness_control"] = {"mAP_50_95": 0.12345678901234568,
                                             "mAP_50": 0.23456789012345678,
                                             "coco_stats": {"AP_large": None}}
        for metrics in methods.values():
            metrics["n_images"] = 2
    for correlation in data["corr_summary"].values():
        correlation.update(valid_count=2, undefined_count=0)
    manifest = {
        "experiment": "02", "run_kind": "full", "seed": 42,
        "conditions": list(formatter.CONDITIONS), "methods": list(formatter.METHODS),
        "preprocessing": {key: data["params"][key] for key in formatter.PARAMETERS[:5]},
        "dataset": {"all_images": True, "expected_images": 2, "evaluated_images": 2,
                    "images": [{"id": 1}, {"id": 2}]},
    }
    del data["params"]
    return data, manifest, {"status": "complete", "inference_calls": 48}


class CompactSummaryTests(unittest.TestCase):
    def test_legacy_summary_keeps_exact_values_without_fabricating_fourth_method(self):
        for name in ARCHIVE_HASHES:
            with self.subTest(experiment=name):
                data = legacy_result(name)
                self.assertEqual(formatter.compact_summary(data), data)
                self.assertEqual(set(data), {"map_results", "corr_summary", "params"})
                for methods in data["map_results"].values():
                    self.assertNotIn("ad_brightness_control", methods)

    def test_detailed_projection_preserves_precision_and_all_four_methods(self):
        data, manifest, _ = complete_fixture()
        original_data, original_manifest = deepcopy(data), deepcopy(manifest)
        summary = formatter.compact_summary(data, manifest)
        self.assertEqual(set(summary), {"map_results", "corr_summary", "params"})
        self.assertEqual(tuple(summary["params"]), formatter.PARAMETERS)
        self.assertEqual(summary["params"]["N_IMAGES"], 2)
        self.assertEqual(summary["params"]["RANDOM_SEED"], 42)
        for condition in formatter.CONDITIONS:
            self.assertEqual(tuple(summary["map_results"][condition]), formatter.METHODS)
            for method in formatter.METHODS:
                metrics = summary["map_results"][condition][method]
                self.assertEqual(set(metrics), set(formatter.METRICS))
                for metric in formatter.METRICS:
                    self.assertEqual(metrics[metric], data["map_results"][condition][method][metric])
            self.assertEqual(set(summary["corr_summary"][condition]), {"mean", "std"})
        encoded = json.dumps(summary, allow_nan=False)
        self.assertIn("0.12345678901234568", encoded)
        self.assertEqual(json.loads(encoded), summary)
        self.assertEqual(data, original_data)
        self.assertEqual(manifest, original_manifest)

    def test_null_metrics_and_correlations_remain_undefined(self):
        data, manifest, _ = complete_fixture()
        data["map_results"]["clean"]["ad_log"]["mAP_50"] = None
        data["corr_summary"]["clean"].update(mean=None, std=None)
        summary = formatter.compact_summary(data, manifest)
        self.assertIsNone(summary["map_results"]["clean"]["ad_log"]["mAP_50"])
        self.assertEqual(summary["corr_summary"]["clean"], {"mean": None, "std": None})
        report = formatter.render_report("02", summary, "runs/full/results.json")
        self.assertIn("| `clean` | N/A | N/A |", report)
        self.assertIn("| N/A |", report)
        self.assertNotIn("| nan |", report.lower())

    def test_nonfinite_metric_values_are_rejected(self):
        for value in [float("nan"), float("inf"), float("-inf")]:
            data, manifest, _ = complete_fixture()
            data["map_results"]["clean"]["baseline"]["mAP_50"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                formatter.compact_summary(data, manifest)

    def test_inconsistent_conditions_and_methods_are_rejected(self):
        data = legacy_result()
        del data["map_results"]["haze"]
        with self.assertRaises(ValueError):
            formatter.compact_summary(data)
        data = legacy_result()
        data["map_results"]["clean"]["ad_brightness_control"] = deepcopy(
            data["map_results"]["clean"]["baseline"])
        with self.assertRaises(ValueError):
            formatter.compact_summary(data)

    def test_reports_share_columns_order_precision_and_comparability_warning(self):
        data, manifest, _ = complete_fixture()
        summaries = [(name, formatter.compact_summary(legacy_result(name)))
                     for name in ARCHIVE_HASHES]
        summaries.append(("02", formatter.compact_summary(data, manifest)))
        header = "| Condition | Baseline | AD only | AD + LoG | AD + brightness control |"
        for name, summary in summaries:
            experiment_id = f"01/runs/{name}" if name in ARCHIVE_HASHES else name
            report = formatter.render_report(experiment_id, summary, "results.json")
            with self.subTest(experiment=name):
                report.encode("ascii")
                self.assertEqual(report.count(header), 2)
                rows = [line for line in report.splitlines() if line.startswith("| `")]
                conditions = [line.split("`")[1] for line in rows[:18]]
                self.assertEqual(conditions, list(formatter.CONDITIONS) * 3)
                self.assertIn("directly comparable", report)
                self.assertIn("fractions, not percentages", report)
                if name in ARCHIVE_HASHES:
                    self.assertTrue(all(line.endswith("| Not run |") for line in rows[:12]))
                    self.assertIn("incorrect class mapping", report)
                else:
                    self.assertIn("0.123457", report)
                    self.assertNotIn("0.12345678901234568", report)
                    self.assertTrue(all("Not run" not in line for line in rows[:12]))


class BuildOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "experiments/02/runs/full_fixture"
        self.run_dir.mkdir(parents=True)
        self.data, self.manifest, self.status = complete_fixture()
        for name in ARCHIVE_HASHES:
            target = legacy_dir(self.root, name) / "results.json"
            target.parent.mkdir(parents=True)
            target.write_bytes((legacy_dir(ROOT, name) / "results.json").read_bytes())
        self.write_run()

    def write_run(self):
        for filename, data in [("results.json", self.data), ("manifest.json", self.manifest),
                               ("status.json", self.status)]:
            (self.run_dir / filename).write_text(json.dumps(data), encoding="utf-8")

    def test_output_set_is_exact_and_sources_are_untouched_without_raw_predictions(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        outputs = formatter.build_outputs(self.run_dir, root=self.root)
        directories = [legacy_dir(self.root, name) for name in ARCHIVE_HASHES] + [self.run_dir]
        expected = {directory / filename for directory in directories
                    for filename in ("RESULTS.md", "summary.json")}
        self.assertEqual(set(outputs), expected)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*.json")}, before)
        self.assertFalse((self.run_dir / "detections").exists())
        self.assertTrue(all(not path.exists() for path in outputs))
        current = json.loads(outputs[self.run_dir / "summary.json"])
        self.assertEqual(current, formatter.compact_summary(self.data, self.manifest))

    def test_incomplete_smoke_and_inconsistent_full_runs_are_rejected(self):
        variants = [
            ("status", "status", "failed"), ("data", "run_kind", "smoke"),
            ("manifest", "run_kind", "smoke"), ("manifest", "experiment", "other"),
            ("data", "image_count", 1), ("data", "inference_calls", 47),
            ("status", "inference_calls", 47),
        ]
        for location, key, value in variants:
            self.data, self.manifest, self.status = complete_fixture()
            getattr(self, location)[key] = value
            self.write_run()
            with self.subTest(location=location, key=key), self.assertRaises(ValueError):
                formatter.build_outputs(self.run_dir, root=self.root)
        for key, value in [("all_images", False), ("expected_images", 3),
                           ("evaluated_images", 1), ("images", [{"id": 1}, {"id": 1}])]:
            self.data, self.manifest, self.status = complete_fixture()
            self.manifest["dataset"][key] = value
            self.write_run()
            with self.subTest(dataset_key=key), self.assertRaises(ValueError):
                formatter.build_outputs(self.run_dir, root=self.root)

    def test_corrected_run_requires_brightness_control(self):
        for methods in self.data["map_results"].values():
            del methods["ad_brightness_control"]
        self.write_run()
        with self.assertRaises(ValueError):
            formatter.build_outputs(self.run_dir, root=self.root)

    def test_current_and_archived_manifest_identifiers_are_supported_without_rewriting(self):
        for identifier in ("02", "003_corrected_full"):
            with self.subTest(identifier=identifier):
                self.manifest["experiment"] = identifier
                self.write_run()
                before = (self.run_dir / "manifest.json").read_bytes()
                outputs = formatter.build_outputs(self.run_dir, root=self.root)
                self.assertIn("`02/runs/full_fixture`", outputs[self.run_dir / "RESULTS.md"])
                self.assertEqual((self.run_dir / "manifest.json").read_bytes(), before)

    def test_read_json_rejects_nonfinite_constants(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            path = self.run_dir / "invalid.json"
            path.write_text('{"value": ' + value + '}', encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                formatter.read_json(path)

    def test_check_mode_detects_missing_views_without_writing_then_passes(self):
        outputs = formatter.build_outputs(self.run_dir, root=self.root)
        with patch.object(formatter, "build_outputs", return_value=outputs), \
                patch.object(formatter, "ROOT", self.root), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                formatter.main(["--run-dir", str(self.run_dir), "--check"])
            self.assertEqual(error.exception.code, 1)
            self.assertTrue(all(not path.exists() for path in outputs))
            for path, text in outputs.items():
                path.write_text(text, encoding="utf-8")
            before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
            formatter.main(["--run-dir", str(self.run_dir), "--check"])
            self.assertEqual({path: path.read_bytes() for path in before}, before)


class RecordedResultTests(unittest.TestCase):
    def test_archived_result_fingerprints_are_preserved(self):
        for name, expected_hash in ARCHIVE_HASHES.items():
            path = legacy_dir(ROOT, name) / "results.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)

    def test_real_full_result_is_unchanged_and_published_views_match(self):
        if not all((FULL_RUN / name).is_file() for name in ("results.json", "manifest.json", "status.json")):
            self.skipTest("Optional detailed full-run metadata is not included in this checkout")
        source = FULL_RUN / "results.json"
        before = source.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),
                         "fb30fd1aadba2cb9de8f46a865f24583f8a356e33da2839a573b269cb7b2d830")
        outputs = formatter.build_outputs(FULL_RUN)
        self.assertEqual(source.read_bytes(), before)
        for path, expected in outputs.items():
            with self.subTest(path=path):
                self.assertEqual(path.read_text(encoding="utf-8"), expected)

    def test_formatter_import_needs_no_ml_dependencies_or_inference(self):
        original_import = builtins.__import__

        def no_ml_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "cv2", "numpy", "ultralytics", "pycocotools", "scipy"}:
                raise AssertionError(f"Formatting attempted an ML import: {name}")
            return original_import(name, *args, **kwargs)

        fresh = importlib.util.module_from_spec(SPEC)
        with patch("builtins.__import__", side_effect=no_ml_import):
            SPEC.loader.exec_module(fresh)
            self.assertEqual(fresh.compact_summary(legacy_result()), legacy_result())


if __name__ == "__main__":
    unittest.main()
