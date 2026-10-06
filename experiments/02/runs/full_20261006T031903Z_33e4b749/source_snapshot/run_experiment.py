"""Corrected, paired AD/LoG experiment; full COCO val2017 by default."""

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import cv2
import numpy as np
from pycocotools.coco import COCO

from evaluation import (
    COCO_EVALUATION_OPTIONS, DETECTOR_OPTIONS, build_category_mapping,
    detect_image, evaluate_detections,
)
from preprocessing import (
    AD_GAMMA, AD_KAPPA, AD_NITER, BLUR_KERNEL, CONDITIONS,
    GAUSSIAN_NOISE_SIGMA, HAZE_A, HAZE_BETA, LOG_BETA, LOG_SIGMA,
    LOW_LIGHT_GAMMA, METHODS, degrade_image, prepare_methods,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
COCO_IMG_DIR = PROJECT_ROOT / "shared/coco/images/val2017"
COCO_ANN_FILE = PROJECT_ROOT / "shared/coco/annotations/instances_val2017.json"
MODEL_PATH = PROJECT_ROOT / "shared/yolo11n.pt"
EXPECTED_IMAGES = 5000
RANDOM_SEED = 42
SMOKE_IMAGES = 3
SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def public_text(value):
    return str(value).replace(str(PROJECT_ROOT), "<PROJECT_ROOT>").replace(
        str(Path.home()), "<HOME>")


class PublicTee:
    """Keep an ordinary text log without private home-directory prefixes."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        safe = public_text(text)
        self.console.write(safe)
        self.log.write(safe)
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    # Exclusive creation protects both completed and partially written artifacts.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")


def write_status(directory, value):
    temporary = directory / f".status-{uuid.uuid4().hex}.tmp"
    try:
        write_json(temporary, value)
        os.replace(temporary, directory / "status.json")
    finally:
        temporary.unlink(missing_ok=True)


def read_image(image_dir, record, verify_hash=False):
    filename = record["file_name"]
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe annotation image name: {filename}")
    path = (Path(image_dir) / relative).resolve()
    if not path.is_relative_to(Path(image_dir).resolve()):
        raise ValueError(f"Image escapes dataset directory: {filename}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify_hash and digest != record["sha256"]:
        raise ValueError(f"Image changed after preflight: {filename}")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {filename}")
    if image.shape[:2] != (record["height"], record["width"]):
        raise ValueError(f"Image/annotation dimensions differ: {filename}")
    return image, digest


def validate_dataset(coco_gt, image_dir, expected_images=EXPECTED_IMAGES,
                     smoke_test=False, seed=RANDOM_SEED):
    records = coco_gt.dataset["images"]
    ids = [item["id"] for item in records]
    names = [item["file_name"] for item in records]
    if len(ids) != expected_images or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {expected_images} unique annotated images")
    if len(set(names)) != len(names):
        raise ValueError("Annotation image filenames must be unique")
    ordered = sorted(records, key=lambda item: item["id"])
    if smoke_test:
        rng = np.random.default_rng(seed)
        positions = sorted(rng.choice(len(ordered), min(SMOKE_IMAGES, len(ordered)),
                                      replace=False).tolist())
        ordered = [ordered[index] for index in positions]
    manifest = []
    for index, record in enumerate(ordered, 1):
        image, digest = read_image(image_dir, record)
        manifest.append({"id": int(record["id"]), "file_name": record["file_name"],
                         "height": int(image.shape[0]), "width": int(image.shape[1]),
                         "sha256": digest})
        if index % 500 == 0 or index == len(ordered):
            print(f"Preflight: {index}/{len(ordered)} images checked", flush=True)
    return manifest


def seed_runtime(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    cv2.setNumThreads(1)


def environment_info():
    import torch

    packages = {dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    return {"python": platform.python_version(), "system": platform.system(),
            "system_release": platform.release(), "machine": platform.machine(),
            "packages": dict(sorted(packages.items())),
            "torch_version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def git_info():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL))
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def source_snapshot(directory):
    target = directory / "source_snapshot"
    target.mkdir()
    hashes = {}
    for path in sorted(EXPERIMENT_DIR.glob("*.py")) + [EXPERIMENT_DIR / "README.md"]:
        shutil.copyfile(path, target / path.name)
        hashes[path.name] = sha256_file(target / path.name)
    return hashes


def prepare_inputs(image_dir, annotation_file, model_path, seed, smoke_test,
                   expected_images, model_factory=None):
    if not Path(model_path).is_file():
        raise FileNotFoundError("Local YOLO weights are missing; no download attempted")
    annotation_hash = sha256_file(annotation_file)
    model_hash = sha256_file(model_path)
    coco_gt = COCO(str(annotation_file))
    if sha256_file(annotation_file) != annotation_hash:
        raise ValueError("Annotations changed while loading")
    records = validate_dataset(coco_gt, image_dir, expected_images, smoke_test, seed)
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(model_path))
    if model.task != "detect":
        raise ValueError("A COCO object-detection model is required")
    mapping = build_category_mapping(model.names, coco_gt)
    model.fuse()
    if sha256_file(model_path) != model_hash:
        raise ValueError("Weights changed while loading")
    return coco_gt, model, records, mapping, annotation_hash, model_hash


def load_detections(path, image_ids):
    detections, seen = [], []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            seen.append(row["image_id"])
            if any(item["image_id"] != row["image_id"] for item in row["detections"]):
                raise ValueError("Detection and record image IDs differ")
            detections.extend(row["detections"])
    if seen != image_ids:
        raise ValueError("Prediction records do not match the evaluated image sequence")
    return detections


def summarize_correlations(rows):
    values = [row["rho_log_g"] for row in rows if row["rho_log_g"] is not None]
    reasons = {}
    for row in rows:
        reason = row["correlation_status"]
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "valid_count": len(values), "undefined_count": len(rows) - len(values),
            "status_counts": reasons}


def primary_deltas(metrics):
    comparisons = {"ad_log_minus_ad_only": ("ad_log", "ad_only"),
                   "ad_log_minus_brightness_control": ("ad_log", "ad_brightness_control"),
                   "ad_only_minus_baseline": ("ad_only", "baseline")}
    result = {}
    for name, (left, right) in comparisons.items():
        result[name] = {}
        for metric in ("mAP_50_95", "mAP_50"):
            a, b = metrics[left][metric], metrics[right][metric]
            result[name][metric] = a - b if a is not None and b is not None else None
    return result


def create_run_directory(output_dir, smoke_test):
    if output_dir is None:
        kind = "smoke" if smoke_test else "full"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = EXPERIMENT_DIR / "runs" / f"{kind}_{stamp}_{uuid.uuid4().hex[:8]}"
    directory = Path(output_dir).absolute()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    return directory


def execute_run(directory, image_dir, annotation_file, model_path, seed, device,
                smoke_test, expected_images, model_factory, starting_git):
    started = time.perf_counter()
    state = {"status": "preflight", "started_utc": utc_now(), "processed_images": 0,
             "condition": None, "last_image_id": None, "inference_calls": 0}
    write_status(directory, state)
    try:
        sources = source_snapshot(directory)
        seed_runtime(seed)
        coco_gt, model, records, mapping, ann_hash, model_hash = prepare_inputs(
            image_dir, annotation_file, model_path, seed, smoke_test, expected_images,
            model_factory)
        image_ids = [record["id"] for record in records]
        options = {**DETECTOR_OPTIONS, "device": device}
        manifest = {
            "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT_DIR.name,
            "run_kind": "smoke" if smoke_test else "full", "started_utc": state["started_utc"],
            "seed": seed, "seed_scheme": "NumPy SeedSequence([seed, condition_index, image_id])",
            "conditions": list(CONDITIONS), "methods": list(METHODS),
            "dataset": {"name": "COCO val2017", "expected_images": expected_images,
                        "evaluated_images": len(records), "all_images": not smoke_test,
                        "annotation_sha256": ann_hash, "images": records},
            "model": {"name": Path(model_path).name, "sha256": model_hash,
                      "task": model.task, "fused": True, "category_mapping": mapping},
            "detector_options": options, "environment": environment_info(),
            "evaluation_options": COCO_EVALUATION_OPTIONS,
            "preprocessing": {"AD_KAPPA": AD_KAPPA, "AD_NITER": AD_NITER,
                              "AD_GAMMA": AD_GAMMA, "AD_BOUNDARY": "no_flux",
                              "LOG_SIGMA": LOG_SIGMA, "LOG_BETA": LOG_BETA,
                              "LOW_LIGHT_GAMMA": LOW_LIGHT_GAMMA,
                              "GAUSSIAN_NOISE_SIGMA": GAUSSIAN_NOISE_SIGMA,
                              "BLUR_KERNEL": list(BLUR_KERNEL), "BLUR_SIGMA": 0,
                              "HAZE_BETA": HAZE_BETA, "HAZE_A": HAZE_A,
                              "HAZE_DEPTH_RANGE": [0.1, 1.0],
                              "BRIGHTNESS_CONTROL": "uniform_gain_match_post_uint8_mean_BGR",
                              "BRIGHTNESS_GAIN_RANGE": [1.0, 1.0 + LOG_BETA],
                              "SATURATION_FRACTION": "channel_values_equal_to_255"},
            "source_sha256": sources, "git": starting_git,
            "raw_detection_format": "gzip JSONL: one image_id/detections record per selected image",
            "primary_comparisons": ["ad_log_minus_ad_only", "ad_log_minus_brightness_control"],
        }
        write_json(directory / "manifest.json", manifest)
        (directory / "detections").mkdir()
        (directory / "diagnostics").mkdir()
        (directory / "metrics").mkdir()
        all_metrics, all_correlations, all_deltas = {}, {}, {}
        effective_saved = False
        for condition_index, condition in enumerate(CONDITIONS):
            prediction_dir = directory / "detections" / condition
            prediction_dir.mkdir()
            state.update(status="inference", condition=condition, processed_images=0,
                         last_image_id=None)
            write_status(directory, state)
            diagnostics = []
            with ExitStack() as stack:
                streams = {method: stack.enter_context(gzip.open(
                    prediction_dir / f"{method}.jsonl.gz", "xt", encoding="utf-8"))
                    for method in METHODS}
                diagnostic_stream = stack.enter_context((directory / "diagnostics" /
                    f"{condition}.jsonl").open("x", encoding="utf-8"))
                for index, record in enumerate(records, 1):
                    image_id = record["id"]
                    state["current_image_id"] = image_id
                    image, _ = read_image(image_dir, record, verify_hash=True)
                    rng = np.random.default_rng(np.random.SeedSequence([seed, condition_index, image_id]))
                    degraded = degrade_image(image, condition, rng)
                    variants, diagnostic = prepare_methods(degraded)
                    if set(variants) != set(METHODS):
                        raise ValueError("Unexpected preprocessing method set")
                    for method in METHODS:
                        state["current_method"] = method
                        detections = detect_image(model, variants[method], mapping, options)
                        for item in detections:
                            item["image_id"] = image_id
                        row = {"image_id": image_id, "detections": detections}
                        streams[method].write(json.dumps(row, allow_nan=False) + "\n")
                        streams[method].flush()
                        state["inference_calls"] += 1
                    if not effective_saved:
                        predictor = getattr(model, "predictor", None)
                        args = getattr(predictor, "args", None)
                        effective = {key: getattr(args, key, value) for key, value in options.items()}
                        backend = getattr(predictor, "model", None)
                        write_json(directory / "effective_detector.json", {
                            "options": effective, "environment": environment_info(),
                            "backend_device": str(getattr(backend, "device", device)),
                            "backend_fp16": bool(getattr(backend, "fp16", False)),
                            "quantize": getattr(args, "quantize", None)})
                        effective_saved = True
                    diagnostic.update(image_id=image_id)
                    diagnostic_stream.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                    diagnostic_stream.flush()
                    diagnostics.append(diagnostic)
                    state.update(processed_images=index, last_image_id=image_id)
                    write_status(directory, state)
                    if index % 100 == 0 or index == len(records):
                        print(f"{condition}: {index}/{len(records)} images", flush=True)
            state["status"] = "evaluation"
            write_status(directory, state)
            condition_metrics = {}
            for method in METHODS:
                state["current_method"] = method
                write_status(directory, state)
                detections = load_detections(prediction_dir / f"{method}.jsonl.gz", image_ids)
                metrics = evaluate_detections(coco_gt, detections, image_ids)
                write_json(directory / "metrics" / f"{condition}_{method}.json", metrics)
                condition_metrics[method] = metrics
                del detections
            all_metrics[condition] = condition_metrics
            all_correlations[condition] = summarize_correlations(diagnostics)
            all_deltas[condition] = primary_deltas(condition_metrics)
        artifacts = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name not in {"run.log", "status.json"}:
                artifacts[path.relative_to(directory).as_posix()] = sha256_file(path)
        output = {"schema_version": SCHEMA_VERSION, "run_kind": manifest["run_kind"],
                  "image_count": len(records), "map_results": all_metrics,
                  "corr_summary": all_correlations, "paired_deltas": all_deltas,
                  "inference_calls": state["inference_calls"],
                  "artifact_sha256": artifacts, "elapsed_seconds": time.perf_counter() - started,
                  "finished_utc": utc_now()}
        write_json(directory / "results.json", output)
        state.update(status="complete", finished_utc=output["finished_utc"],
                     elapsed_seconds=output["elapsed_seconds"])
        write_status(directory, state)
        print(f"Completed {manifest['run_kind']} run: {len(records)} images, "
              f"{state['inference_calls']} detector calls", flush=True)
        print(f"Artifacts: {public_text(directory)}", flush=True)
        return directory
    except BaseException as error:
        state.update(status="failed", failed_utc=utc_now(), error_type=type(error).__name__,
                     error=public_text(error), elapsed_seconds=time.perf_counter() - started)
        write_status(directory, state)
        traceback.print_exc()
        raise


def run_experiment(*, image_dir=COCO_IMG_DIR, annotation_file=COCO_ANN_FILE,
                   model_path=MODEL_PATH, output_dir=None, seed=RANDOM_SEED,
                   device="cpu", smoke_test=False, check_only=False,
                   expected_images=EXPECTED_IMAGES, model_factory=None):
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    if check_only:
        seed_runtime(seed)
        _, _, records, mapping, ann_hash, model_hash = prepare_inputs(
            image_dir, annotation_file, model_path, seed, False, expected_images, model_factory)
        report = {"status": "preflight_passed", "image_count": len(records),
                  "category_count": len(mapping), "annotation_sha256": ann_hash,
                  "model_sha256": model_hash, "inference_performed": False}
        print(json.dumps(report, indent=2))
        return report
    starting_git = git_info()
    directory = create_run_directory(output_dir, smoke_test)
    with (directory / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(PublicTee(sys.stdout, log)), redirect_stderr(PublicTee(sys.stderr, log)):
            return execute_run(directory, image_dir, annotation_file, model_path, seed,
                               device, smoke_test, expected_images, model_factory, starting_git)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate all inputs without inference")
    mode.add_argument("--smoke-test", action="store_true", help="Technical check on three images, not a full result")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, help="New output directory; must not already exist")
    arguments = parser.parse_args(argv)
    run_experiment(seed=arguments.seed, device=arguments.device,
                   output_dir=arguments.output_dir, check_only=arguments.check_only,
                   smoke_test=arguments.smoke_test)


if __name__ == "__main__":
    main()
