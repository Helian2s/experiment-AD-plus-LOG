"""Explicit detector settings and validated COCO bounding-box evaluation."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DETECTOR_OPTIONS = {
    "conf": 0.001,
    "iou": 0.7,
    "imgsz": 640,
    "max_det": 300,
    "device": "cpu",
    "quantize": 32,
    "augment": False,
    "agnostic_nms": False,
    "verbose": False,
    "save": False,
    "rect": True,
    "batch": 1,
    "classes": None,
}
EVALUATION_MAX_DETS = [1, 10, 100]
COCO_EVALUATION_OPTIONS = {
    "iou_type": "bbox",
    "max_dets": EVALUATION_MAX_DETS,
    "iou_thresholds": np.linspace(0.5, 0.95, 10).tolist(),
    "recall_thresholds": np.linspace(0, 1, 101).tolist(),
    "area_ranges": [[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e10]],
    "area_labels": ["all", "small", "medium", "large"],
    "use_categories": True,
}
COCO_STAT_NAMES = (
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return int(value)


def build_category_mapping(model_names, coco_gt):
    """Match all 80 contiguous model classes to COCO's noncontiguous IDs."""
    if isinstance(model_names, Mapping):
        names = {_integer(key, "Model class ID"): value
                 for key, value in model_names.items()}
    elif isinstance(model_names, Sequence) and not isinstance(model_names, str):
        names = dict(enumerate(model_names))
    else:
        raise ValueError("Model names must be a mapping or sequence")
    if set(names) != set(range(80)):
        raise ValueError("Expected exactly 80 model classes numbered 0 through 79")
    if any(not isinstance(name, str) or not name for name in names.values()):
        raise ValueError("Model class names must be nonempty strings")
    if len(set(names.values())) != 80:
        raise ValueError("Model class names must be unique")

    categories = coco_gt.dataset["categories"]
    if len(categories) != 80:
        raise ValueError("Expected exactly 80 COCO categories")
    category_ids = [_integer(category["id"], "COCO category ID")
                    for category in categories]
    category_names = [category["name"] for category in categories]
    if len(set(category_ids)) != 80 or min(category_ids) < 1:
        raise ValueError("COCO category IDs must be unique positive integers")
    if any(not isinstance(name, str) or not name for name in category_names):
        raise ValueError("COCO category names must be nonempty strings")
    if len(set(category_names)) != 80:
        raise ValueError("COCO category names must be unique")
    if set(names.values()) != set(category_names):
        missing = sorted(set(names.values()) - set(category_names))
        extra = sorted(set(category_names) - set(names.values()))
        raise ValueError(f"Model/COCO category mismatch: missing={missing}, extra={extra}")
    ids_by_name = dict(zip(category_names, category_ids))
    return {index: ids_by_name[names[index]] for index in range(80)}


def _bbox_and_score(bbox, score):
    coordinates = np.asarray(bbox, dtype=np.float64)
    confidence = float(score)
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("Detection bbox must contain four finite numbers")
    # Boundary clipping can produce zero-area boxes. Keep them for COCOeval
    # instead of removing potential false positives or aborting a long run.
    if coordinates[2] < 0 or coordinates[3] < 0:
        raise ValueError("Detection bbox width and height must be nonnegative")
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Detection score must be finite and between zero and one")
    return coordinates.tolist(), confidence


def detect_image(model, image, category_mapping, options=None):
    """Return COCO-format predictions without image IDs; reject invalid output."""
    settings = dict(DETECTOR_OPTIONS)
    if options is not None:
        unknown = set(options) - set(settings)
        if unknown:
            raise ValueError(f"Unsupported detector options: {sorted(unknown)}")
        settings.update(options)
    results = model.predict(source=image, **settings)
    if len(results) != 1:
        raise ValueError("Expected exactly one detection result for one image")
    if results[0].boxes is None:
        raise ValueError("Detector did not return bounding-box results")
    detections = []
    for box in results[0].boxes:
        xyxy = np.asarray(box.xyxy[0].tolist(), dtype=np.float64)
        if xyxy.shape != (4,) or not np.isfinite(xyxy).all():
            raise ValueError("Detector returned nonfinite or malformed coordinates")
        raw_class = float(box.cls[0])
        if not np.isfinite(raw_class) or not raw_class.is_integer():
            raise ValueError("Detector returned an invalid class index")
        class_index = int(raw_class)
        if class_index not in category_mapping:
            raise ValueError(f"Detector class {class_index} has no COCO mapping")
        x1, y1, x2, y2 = xyxy
        bbox, score = _bbox_and_score([x1, y1, x2 - x1, y2 - y1], box.conf[0])
        detections.append({
            "bbox": bbox,
            "score": score,
            "category_id": _integer(category_mapping[class_index], "COCO category ID"),
        })
    return detections


def evaluate_detections(coco_gt, detections, image_ids):
    """Evaluate exactly the declared image set; never turn failures into scores."""
    ids = [_integer(image_id, "Evaluation image ID") for image_id in image_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation image IDs must be nonempty and unique")
    id_set = set(ids)
    unknown_ids = id_set - set(coco_gt.getImgIds())
    if unknown_ids:
        raise ValueError(f"Unknown evaluation image IDs: {sorted(unknown_ids)}")
    category_ids = set(coco_gt.getCatIds())
    normalized = []
    for detection in detections:
        image_id = _integer(detection["image_id"], "Detection image ID")
        category_id = _integer(detection["category_id"], "Detection category ID")
        if image_id not in id_set:
            raise ValueError(f"Detection image {image_id} is outside the evaluation set")
        if category_id not in category_ids:
            raise ValueError(f"Unknown detection category: {category_id}")
        bbox, score = _bbox_and_score(detection["bbox"], detection["score"])
        normalized.append({"image_id": image_id, "category_id": category_id,
                           "bbox": bbox, "score": score})

    if normalized:
        coco_dt = coco_gt.loadRes(normalized)
    else:
        # COCO.loadRes indexes the first annotation and cannot load an empty list.
        coco_dt = COCO()
        coco_dt.dataset = {
            "info": deepcopy(coco_gt.dataset.get("info", {})),
            "images": deepcopy(coco_gt.dataset["images"]),
            "categories": deepcopy(coco_gt.dataset["categories"]),
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(ids)
    evaluator.params.catIds = sorted(category_ids)
    evaluator.params.maxDets = list(EVALUATION_MAX_DETS)
    evaluator.params.iouThrs = np.array(COCO_EVALUATION_OPTIONS["iou_thresholds"])
    evaluator.params.recThrs = np.array(COCO_EVALUATION_OPTIONS["recall_thresholds"])
    evaluator.params.areaRng = deepcopy(COCO_EVALUATION_OPTIONS["area_ranges"])
    evaluator.params.areaRngLbl = list(COCO_EVALUATION_OPTIONS["area_labels"])
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    statistics = np.asarray(evaluator.stats, dtype=np.float64)
    if statistics.shape != (12,) or not np.isfinite(statistics).all():
        raise ValueError("COCO evaluation returned malformed or nonfinite statistics")
    if np.any((statistics < 0) & (statistics != -1)) or np.any(statistics > 1):
        raise ValueError("COCO evaluation returned out-of-range statistics")
    metrics = {name: None if value == -1 else float(value)
               for name, value in zip(COCO_STAT_NAMES, statistics)}
    return {
        "mAP_50_95": metrics["AP"],
        "mAP_50": metrics["AP50"],
        "coco_stats": metrics,
        "n_images": len(ids),
        "n_detections": len(normalized),
        "n_zero_area_detections": sum(item["bbox"][2] == 0 or item["bbox"][3] == 0
                                      for item in normalized),
        "n_images_with_detections": len({item["image_id"] for item in normalized}),
        "n_ground_truth_annotations": len(coco_gt.getAnnIds(imgIds=ids)),
        "evaluation": {
            **deepcopy(COCO_EVALUATION_OPTIONS),
            "image_ids": sorted(ids),
            "category_ids": sorted(category_ids),
        },
    }
