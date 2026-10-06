"""End-to-end artifact checks on tiny data with no real detector inference."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/003_corrected_full"
SPEC = importlib.util.spec_from_file_location(
    "corrected_runner", EXPERIMENT / "run_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(EXPERIMENT))
try:
    SPEC.loader.exec_module(runner)
finally:
    sys.path.remove(str(EXPERIMENT))


def reject_nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def read_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=reject_nonfinite) for line in stream]


class FakeDetector:
    task = "detect"
    names = {index: f"class_{index}" for index in range(80)}

    def __init__(self, empty=False):
        self.empty = empty
        self.calls = []
        self.fused = False
        self.predictor = None

    def fuse(self):
        self.fused = True

    def predict(self, *, source, **options):
        self.calls.append({"sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                           "options": options})
        self.predictor = SimpleNamespace(args=SimpleNamespace(**options))
        boxes = [] if self.empty else [SimpleNamespace(
            xyxy=np.array([[2., 3., 6., 8.]]), conf=np.array([0.9]), cls=np.array([11.]))]
        return [SimpleNamespace(boxes=boxes)]


class CorrectedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.image_dir = self.temporary / "images"
        self.image_dir.mkdir()
        self.annotation_file = self.temporary / "instances.json"
        self.model_path = self.temporary / "fake-weights.pt"
        self.model_path.write_bytes(b"Fake detector fixture, never loaded as torch weights")
        self.output = self.temporary / "run"
        self.models = []
        self.empty_predictions = False
        self.dataset = {
            "images": [{"id": image_id, "file_name": f"{image_id}.png",
                        "height": 16, "width": 20} for image_id in [2, 1]],
            "categories": [{"id": category_id, "name": f"class_{index}"}
                           for index, category_id in enumerate(
                               list(range(1, 12)) + list(range(13, 82)))],
            "annotations": [{"id": image_id, "image_id": image_id, "category_id": 13,
                             "bbox": [2, 3, 4, 5], "area": 20, "iscrowd": 0}
                            for image_id in [1, 2]],
        }
        for image_id in [1, 2]:
            image = ((np.arange(16 * 20 * 3).reshape(16, 20, 3) * image_id + 37)
                     % 256).astype(np.uint8)
            self.assertTrue(cv2.imwrite(str(self.image_dir / f"{image_id}.png"), image))
        self.write_annotations()
        self.seed_runtime = self.stack.enter_context(patch.object(runner, "seed_runtime"))
        self.stack.enter_context(patch.object(runner, "environment_info",
                                              return_value={"test_fixture": True}))
        self.stack.enter_context(patch.object(runner, "git_info",
                                              return_value={"revision": None, "dirty": True}))
        self.console = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.console))
        self.stack.enter_context(redirect_stderr(self.console))

    def write_annotations(self):
        self.annotation_file.write_text(json.dumps(self.dataset), encoding="utf-8")

    def model_factory(self, path):
        self.assertEqual(path, str(self.model_path))
        model = FakeDetector(empty=self.empty_predictions)
        self.models.append(model)
        return model

    def run_fixture(self, **overrides):
        arguments = {"image_dir": self.image_dir, "annotation_file": self.annotation_file,
                     "model_path": self.model_path, "output_dir": self.output,
                     "expected_images": 2, "model_factory": self.model_factory}
        arguments.update(overrides)
        return runner.run_experiment(**arguments)

    def assert_failed_run(self, error_type):
        status = read_json(self.output / "status.json")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_type"], error_type)
        self.assertFalse((self.output / "results.json").exists())
        return status

    def test_full_fixture_has_all_metrics_predictions_and_verified_artifacts(self):
        directory = self.run_fixture(seed=87)
        self.assertEqual(directory, self.output)
        result = read_json(directory / "results.json")
        manifest = read_json(directory / "manifest.json")
        status = read_json(directory / "status.json")
        self.assertEqual(status["status"], "complete")
        self.assertEqual(result["run_kind"], "full")
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["inference_calls"], 48)
        self.assertEqual(status["inference_calls"], 48)
        self.assertEqual(len(self.models[0].calls), 48)
        self.assertTrue(self.models[0].fused)
        self.seed_runtime.assert_called_once_with(87)
        self.assertEqual(manifest["seed"], 87)
        self.assertTrue(manifest["dataset"]["all_images"])
        self.assertEqual(manifest["dataset"]["expected_images"], 2)
        self.assertEqual(manifest["dataset"]["evaluated_images"], 2)
        self.assertEqual(manifest["dataset"]["annotation_sha256"],
                         runner.sha256_file(self.annotation_file))
        self.assertEqual(manifest["model"]["sha256"], runner.sha256_file(self.model_path))
        self.assertEqual(manifest["model"]["category_mapping"]["11"], 13)
        self.assertEqual([record["id"] for record in manifest["dataset"]["images"]], [1, 2])
        for record in manifest["dataset"]["images"]:
            self.assertEqual(record["sha256"], runner.sha256_file(
                self.image_dir / record["file_name"]))
        self.assertEqual(len(list((directory / "metrics").glob("*.json"))), 24)
        self.assertEqual(len(list((directory / "detections").glob("*/*.jsonl.gz"))), 24)
        self.assertEqual(set(result["map_results"]), set(runner.CONDITIONS))
        for condition in runner.CONDITIONS:
            self.assertEqual(set(result["map_results"][condition]), set(runner.METHODS))
            for method in runner.METHODS:
                rows = read_predictions(directory / "detections" / condition / f"{method}.jsonl.gz")
                self.assertEqual([row["image_id"] for row in rows], [1, 2])
                for row in rows:
                    self.assertEqual(len(row["detections"]), 1)
                    self.assertEqual(row["detections"][0]["image_id"], row["image_id"])
                    self.assertEqual(row["detections"][0]["category_id"], 13)
                metric = read_json(directory / "metrics" / f"{condition}_{method}.json")
                self.assertEqual(metric, result["map_results"][condition][method])
                self.assertAlmostEqual(metric["mAP_50_95"], 1.)
                self.assertEqual(metric["evaluation"]["image_ids"], [1, 2])
            diagnostics = (directory / "diagnostics" / f"{condition}.jsonl").read_text().splitlines()
            self.assertEqual(len(diagnostics), 2)
            for line in diagnostics:
                json.loads(line, parse_constant=reject_nonfinite)
        for filename, digest in manifest["source_sha256"].items():
            self.assertEqual(digest, runner.sha256_file(directory / "source_snapshot" / filename))
            self.assertEqual(digest, runner.sha256_file(EXPERIMENT / filename))
        for filename, digest in result["artifact_sha256"].items():
            self.assertEqual(digest, runner.sha256_file(directory / filename))
        actual_artifacts = {path.relative_to(directory).as_posix()
                            for path in directory.rglob("*") if path.is_file()
                            and path.name not in {"results.json", "status.json", "run.log"}}
        self.assertEqual(set(result["artifact_sha256"]), actual_artifacts)
        effective = read_json(directory / "effective_detector.json")
        self.assertEqual(effective["options"], manifest["detector_options"])

    def test_all_empty_predictions_complete_with_zero_ap(self):
        self.empty_predictions = True
        self.run_fixture()
        result = read_json(self.output / "results.json")
        for condition in result["map_results"].values():
            for metrics in condition.values():
                self.assertEqual(metrics["mAP_50_95"], 0.)
                self.assertEqual(metrics["n_detections"], 0)
        for path in (self.output / "detections").glob("*/*.jsonl.gz"):
            self.assertEqual(read_predictions(path), [{"image_id": 1, "detections": []},
                                                      {"image_id": 2, "detections": []}])
        self.assertEqual(read_json(self.output / "status.json")["status"], "complete")

    def test_existing_output_is_not_touched(self):
        self.output.mkdir()
        marker = self.output / "results.json"
        marker.write_text("existing result", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(marker.read_text(), "existing result")
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.assertEqual(self.models, [])
        self.seed_runtime.assert_not_called()

    def test_dangling_output_symlink_is_not_overwritten(self):
        missing = self.temporary / "missing"
        self.output.symlink_to(missing, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(missing.exists())
        self.assertEqual(self.models, [])

    def test_unreadable_image_fails_before_inference(self):
        (self.image_dir / "1.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            self.run_fixture()
        self.assert_failed_run("ValueError")
        self.assertEqual(self.models, [])

    def test_missing_image_fails_before_inference(self):
        (self.image_dir / "1.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_fixture()
        self.assert_failed_run("FileNotFoundError")
        self.assertEqual(self.models, [])

    def test_inconsistent_dimensions_fail_before_inference(self):
        self.dataset["images"][1]["width"] = 99
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.run_fixture()
        self.assert_failed_run("ValueError")
        self.assertEqual(self.models, [])

    def test_missing_weights_do_not_trigger_model_factory(self):
        self.model_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no download attempted"):
            self.run_fixture()
        self.assert_failed_run("FileNotFoundError")
        self.assertEqual(self.models, [])

    def test_changed_image_after_preflight_is_rejected(self):
        def changed_after_preflight(path):
            image = cv2.imread(str(self.image_dir / "1.png"))
            self.assertTrue(cv2.imwrite(str(self.image_dir / "1.png"), 255 - image))
            return self.model_factory(path)

        with self.assertRaisesRegex(ValueError, "Image changed after preflight"):
            self.run_fixture(model_factory=changed_after_preflight)
        status = self.assert_failed_run("ValueError")
        self.assertEqual(status["inference_calls"], 0)
        self.assertEqual(self.models[0].calls, [])

    def test_changed_weights_while_loading_are_rejected(self):
        def changed_while_loading(path):
            self.model_path.write_bytes(b"changed weights")
            return self.model_factory(path)

        with self.assertRaisesRegex(ValueError, "Weights changed while loading"):
            self.run_fixture(model_factory=changed_while_loading)
        self.assert_failed_run("ValueError")
        self.assertEqual(self.models[0].calls, [])

    def test_changed_annotations_while_loading_are_rejected(self):
        original_loader = runner.COCO

        def changed_while_loading(path):
            coco = original_loader(path)
            self.annotation_file.write_text(json.dumps(self.dataset) + "\n", encoding="utf-8")
            return coco

        with patch.object(runner, "COCO", side_effect=changed_while_loading):
            with self.assertRaisesRegex(ValueError, "Annotations changed while loading"):
                self.run_fixture()
        self.assert_failed_run("ValueError")
        self.assertEqual(self.models, [])

    def test_evaluation_failure_preserves_predictions_and_marks_failed(self):
        with patch.object(runner, "evaluate_detections",
                          side_effect=RuntimeError("synthetic evaluation failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic evaluation failure"):
                self.run_fixture()
        status = self.assert_failed_run("RuntimeError")
        self.assertEqual(status["inference_calls"], 8)
        self.assertEqual(len(self.models[0].calls), 8)
        self.assertTrue((self.output / "manifest.json").exists())
        for method in runner.METHODS:
            rows = read_predictions(self.output / "detections" / "clean" / f"{method}.jsonl.gz")
            self.assertEqual([row["image_id"] for row in rows], [1, 2])
        self.assertFalse(list((self.output / "metrics").glob("*.json")))
        self.assertIn("synthetic evaluation failure", (self.output / "run.log").read_text())

    def test_check_only_never_infers_or_creates_output(self):
        result = self.run_fixture(check_only=True)
        self.assertEqual(result["status"], "preflight_passed")
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["category_count"], 80)
        self.assertFalse(result["inference_performed"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.models[0].calls, [])
        self.assertTrue(self.models[0].fused)

    def test_seed_repeats_exact_detector_inputs_and_changes_stochastic_conditions(self):
        self.run_fixture(seed=17)
        second = self.temporary / "repeat"
        third = self.temporary / "other-seed"
        self.run_fixture(seed=17, output_dir=second)
        self.run_fixture(seed=18, output_dir=third)
        first_calls, repeated_calls, other_calls = [model.calls for model in self.models]
        self.assertEqual(first_calls, repeated_calls)
        for condition_index, condition in enumerate(runner.CONDITIONS):
            selected = slice(condition_index * 8, (condition_index + 1) * 8)
            if condition in {"gaussian_noise", "multifactor"}:
                self.assertNotEqual(first_calls[selected], other_calls[selected])
            else:
                self.assertEqual(first_calls[selected], other_calls[selected])
        for condition in runner.CONDITIONS:
            self.assertEqual((self.output / "diagnostics" / f"{condition}.jsonl").read_bytes(),
                             (second / "diagnostics" / f"{condition}.jsonl").read_bytes())
            for method in runner.METHODS:
                relative = Path("detections") / condition / f"{method}.jsonl.gz"
                self.assertEqual(read_predictions(self.output / relative),
                                 read_predictions(second / relative))

    def test_invalid_seed_fails_without_output(self):
        for seed in [-1, 2**32]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.run_fixture(seed=seed)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.models, [])

    def test_prediction_loader_rejects_wrong_sequence_or_nested_image_id(self):
        path = self.temporary / "predictions.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"image_id": 2, "detections": []}) + "\n")
            stream.write(json.dumps({"image_id": 1, "detections": []}) + "\n")
        with self.assertRaisesRegex(ValueError, "image sequence"):
            runner.load_detections(path, [1, 2])
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"image_id": 1, "detections": [{"image_id": 2}]}) + "\n")
        with self.assertRaisesRegex(ValueError, "record image IDs differ"):
            runner.load_detections(path, [1])


if __name__ == "__main__":
    unittest.main()
