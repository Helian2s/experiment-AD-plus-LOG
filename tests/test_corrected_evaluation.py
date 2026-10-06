"""Small synthetic tests for corrected detection and COCO evaluation."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pycocotools.coco import COCO


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "corrected_evaluation", ROOT / "experiments/003_corrected_full/evaluation.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
CATEGORY_IDS = list(range(1, 12)) + list(range(13, 82))


def ground_truth(with_annotations=True):
    coco = COCO()
    coco.dataset = {
        "images": [{"id": index, "width": 100, "height": 100}
                   for index in [1, 2]],
        "categories": [{"id": category_id, "name": f"class_{index}"}
                       for index, category_id in enumerate(CATEGORY_IDS)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 13,
             "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 13,
             "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
        ] if with_annotations else [],
    }
    with redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def perfect_detections():
    return [{"image_id": image_id, "category_id": 13,
             "bbox": [coordinate, coordinate, 20, 20], "score": 0.9}
            for image_id, coordinate in [(1, 10), (2, 30)]]


def fake_box(xyxy=(10, 20, 30, 50), score=0.9, class_index=11):
    return SimpleNamespace(xyxy=np.array([xyxy]), conf=np.array([score]),
                           cls=np.array([class_index]))


class CategoryMappingTests(unittest.TestCase):
    def setUp(self):
        self.coco = ground_truth()
        self.names = {index: f"class_{index}" for index in range(80)}

    def test_names_resolve_sparse_category_ids(self):
        self.coco.dataset["categories"].reverse()
        mapping = evaluation.build_category_mapping(self.names, self.coco)
        self.assertEqual(mapping, dict(enumerate(CATEGORY_IDS)))
        self.assertEqual(mapping[11], 13)
        self.assertNotEqual(mapping[11], 11 + 1)

    def test_name_sequence_is_supported(self):
        mapping = evaluation.build_category_mapping(list(self.names.values()), self.coco)
        self.assertEqual(mapping[11], 13)

    def test_invalid_model_classes_are_rejected(self):
        variants = [dict(list(self.names.items())[:-1]),
                    {**self.names, 80: "extra"},
                    {**self.names, 79: "class_0"},
                    {**self.names, 79: "unknown"},
                    {str(key): value for key, value in self.names.items()}]
        for names in variants:
            with self.subTest(names=names), self.assertRaises(ValueError):
                evaluation.build_category_mapping(names, self.coco)

    def test_duplicate_category_names_or_ids_are_rejected(self):
        for field in ["id", "name"]:
            with self.subTest(field=field):
                coco = ground_truth()
                coco.dataset["categories"][1][field] = coco.dataset["categories"][0][field]
                with self.assertRaises(ValueError):
                    evaluation.build_category_mapping(self.names, coco)

    def test_missing_categories_are_rejected(self):
        self.coco.dataset["categories"].pop()
        with self.assertRaises(ValueError):
            evaluation.build_category_mapping(self.names, self.coco)


class DetectionTests(unittest.TestCase):
    def test_explicit_settings_and_sparse_mapping(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[fake_box()])]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = evaluation.detect_image(model, image, {11: 13})
        self.assertEqual(detections, [{"bbox": [10., 20., 20., 30.],
                                     "score": 0.9, "category_id": 13}])
        self.assertIs(model.predict.call_args.kwargs["source"], image)
        expected = {"conf": 0.001, "iou": 0.7, "imgsz": 640, "max_det": 300,
                    "device": "cpu", "quantize": 32, "augment": False,
                    "agnostic_nms": False, "verbose": False, "save": False,
                    "rect": True, "batch": 1, "classes": None}
        self.assertEqual({key: value for key, value in model.predict.call_args.kwargs.items()
                          if key != "source"}, expected)

    def test_options_do_not_mutate_shared_defaults(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[])]
        original = dict(evaluation.DETECTOR_OPTIONS)
        self.assertEqual(evaluation.detect_image(model, None, {}, {"device": "cuda:0"}), [])
        self.assertEqual(model.predict.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(evaluation.DETECTOR_OPTIONS, original)
        with self.assertRaises(ValueError):
            evaluation.detect_image(model, None, {}, {"unknown": True})

    def test_invalid_detector_output_fails_loudly(self):
        boxes = [fake_box(xyxy=(10, 20, np.nan, 50)),
                 fake_box(xyxy=(30, 20, 10, 50)), fake_box(score=np.nan),
                 fake_box(score=1.1), fake_box(class_index=11.5),
                 fake_box(class_index=90)]
        for box in boxes:
            model = Mock()
            model.predict.return_value = [SimpleNamespace(boxes=[box])]
            with self.subTest(box=box), self.assertRaises(ValueError):
                evaluation.detect_image(model, None, {11: 13})

    def test_detector_errors_are_not_hidden(self):
        model = Mock()
        model.predict.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            evaluation.detect_image(model, None, {11: 13})

    def test_missing_or_multiple_detector_results_are_rejected(self):
        for result in [[], [SimpleNamespace(boxes=None)],
                       [SimpleNamespace(boxes=[]), SimpleNamespace(boxes=[])]]:
            model = Mock()
            model.predict.return_value = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                evaluation.detect_image(model, None, {11: 13})


class CocoEvaluationTests(unittest.TestCase):
    def evaluate(self, detections, ids=(1, 2), coco=None):
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate_detections(
                ground_truth() if coco is None else coco, detections, ids)

    def test_perfect_predictions_and_all_twelve_statistics(self):
        detections = perfect_detections()
        unchanged = deepcopy(detections)
        result = self.evaluate(detections)
        self.assertAlmostEqual(result["mAP_50_95"], 1.)
        self.assertAlmostEqual(result["mAP_50"], 1.)
        self.assertEqual(len(result["coco_stats"]), 12)
        self.assertAlmostEqual(result["coco_stats"]["AP_small"], 1.)
        self.assertIsNone(result["coco_stats"]["AP_large"])
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["n_images_with_detections"], 2)
        self.assertEqual(result["n_ground_truth_annotations"], 2)
        self.assertEqual(result["evaluation"]["max_dets"], [1, 10, 100])
        self.assertEqual(detections, unchanged)
        json.dumps(result, allow_nan=False)

    def test_empty_predictions_return_zero_not_scalar(self):
        result = self.evaluate([])
        self.assertEqual(result["mAP_50_95"], 0.)
        self.assertEqual(result["mAP_50"], 0.)
        self.assertEqual(result["coco_stats"]["AR100"], 0.)
        self.assertIsNone(result["coco_stats"]["AP_large"])
        self.assertEqual(result["n_detections"], 0)
        self.assertEqual(result["n_images_with_detections"], 0)
        json.dumps(result, allow_nan=False)

    def test_no_ground_truth_has_undefined_not_zero_metrics(self):
        result = self.evaluate([], coco=ground_truth(with_annotations=False))
        self.assertIsNone(result["mAP_50_95"])
        self.assertTrue(all(value is None for value in result["coco_stats"].values()))

    def test_only_requested_images_are_evaluated(self):
        detections = perfect_detections()[:1]
        result = self.evaluate(detections, ids=[1])
        self.assertAlmostEqual(result["mAP_50_95"], 1.)
        self.assertEqual(result["evaluation"]["image_ids"], [1])
        self.assertEqual(result["n_ground_truth_annotations"], 1)
        result_full = self.evaluate(detections)
        self.assertLess(result_full["mAP_50_95"], 0.6)

    def test_invalid_or_duplicate_image_ids_fail(self):
        for ids in [[], [1, 1], [1, 99], [1.0], [True]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.evaluate([], ids=ids)

    def test_outside_set_and_unknown_category_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(perfect_detections(), ids=[1])
        detections = perfect_detections()
        detections[0]["category_id"] = 12
        with self.assertRaises(ValueError):
            self.evaluate(detections)

    def test_invalid_prediction_geometry_and_scores_are_rejected(self):
        for key, value in [("bbox", [10, 10, -1, 20]), ("score", float("nan")),
                           ("bbox", [10, 10, 20, float("inf")])]:
            detections = perfect_detections()
            detections[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.evaluate(detections)

    def test_zero_area_predictions_are_retained_and_counted(self):
        detections = perfect_detections()
        detections.insert(0, {"image_id": 1, "category_id": 13,
                              "bbox": [0, 0, 0, 20], "score": 1.0})
        result = self.evaluate(detections)
        self.assertEqual(result["n_zero_area_detections"], 1)
        self.assertEqual(result["n_detections"], 3)
        self.assertLess(result["mAP_50_95"], 1.0)

    def test_evaluation_exception_propagates(self):
        with patch.object(evaluation.COCOeval, "evaluate",
                          side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                self.evaluate([])

    def test_malformed_evaluation_statistics_are_rejected(self):
        for statistics in [[0.] * 11, [float("nan")] * 12, [-2.] * 12, [1.1] * 12]:
            def set_statistics(evaluator):
                evaluator.stats = statistics
            with self.subTest(statistics=statistics):
                with patch.object(evaluation.COCOeval, "summarize", set_statistics):
                    with self.assertRaises(ValueError):
                        self.evaluate([])


if __name__ == "__main__":
    unittest.main()
