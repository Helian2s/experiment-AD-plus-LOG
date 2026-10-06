"""Build consistent result summaries without changing experiment evidence."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("clean", "gaussian_noise", "blur", "low_light", "haze", "multifactor")
METHODS = ("baseline", "ad_only", "ad_log", "ad_brightness_control")
METRICS = ("mAP_50_95", "mAP_50")
PARAMETERS = ("AD_KAPPA", "AD_NITER", "AD_GAMMA", "LOG_SIGMA", "LOG_BETA",
              "N_IMAGES", "RANDOM_SEED")
LEGACY_EXPERIMENT = "01"
LEGACY_RUNS = ("pilot_5", "full_5000")
CURRENT_EXPERIMENT = "02"
# Archived launch manifests retain the identifier used before the folder rename.
CURRENT_MANIFEST_IDS = (CURRENT_EXPERIMENT, "003_corrected_full")


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def checked_number(value, minimum, maximum=None, allow_null=True):
    if value is None and allow_null:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid numeric result: {value!r}")
    return value


def compact_summary(data, manifest=None):
    """Project legacy or detailed results onto the shared, lossless scalar view.

    Detailed diagnostics are not discarded from their source. Methods that were
    never measured are omitted, rather than filled with invented zero scores.
    """
    if set(data["map_results"]) != set(CONDITIONS) or set(data["corr_summary"]) != set(CONDITIONS):
        raise ValueError("Results must contain all six conditions")
    methods = set(data["map_results"][CONDITIONS[0]])
    if not set(METHODS[:3]) <= methods or not methods <= set(METHODS):
        raise ValueError("Unexpected or missing experiment methods")
    maps, correlations = {}, {}
    for condition in CONDITIONS:
        source = data["map_results"][condition]
        if set(source) != methods:
            raise ValueError("Method sets differ between conditions")
        maps[condition] = {
            method: {metric: checked_number(source[method][metric], 0, 1)
                     for metric in METRICS}
            for method in METHODS if method in methods
        }
        correlation = data["corr_summary"][condition]
        correlations[condition] = {
            "mean": checked_number(correlation["mean"], -1, 1),
            "std": checked_number(correlation["std"], 0),
        }
    if manifest is None:
        parameters = {key: data["params"][key] for key in PARAMETERS}
    else:
        parameters = {key: manifest["preprocessing"][key] for key in PARAMETERS[:5]}
        parameters.update(N_IMAGES=manifest["dataset"]["evaluated_images"],
                          RANDOM_SEED=manifest["seed"])
    for key, value in parameters.items():
        checked_number(value, 0, allow_null=False)
        if key in {"N_IMAGES", "AD_NITER", "RANDOM_SEED"} and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if parameters["N_IMAGES"] == 0:
        raise ValueError("An experiment summary requires at least one image")
    return {"map_results": maps, "corr_summary": correlations, "params": parameters}


def format_value(value):
    return "N/A" if value is None else f"{value:.6f}"


def render_report(experiment_id, summary, source_relative, protocol_relative="README.md"):
    legacy = experiment_id.split("/")[0] == LEGACY_EXPERIMENT
    lines = [
        "# Experiment Results", "",
        f"- Experiment: `{experiment_id}`",
        f"- Recorded image count: {summary['params']['N_IMAGES']}",
        f"- Random seed: {summary['params']['RANDOM_SEED']}", "",
        "[Machine-readable summary](summary.json) | "
        f"[Source artifact]({source_relative}) | [Protocol and provenance]({protocol_relative})", "",
        "Scores are fractions, not percentages. Tables round to six decimal places;",
        "the JSON retains the recorded precision. `Not run` means the method was",
        "not evaluated; `N/A` means a recorded metric is undefined.", "",
    ]
    if legacy:
        lines.extend([
            "Legacy results retain the incorrect class mapping and brightening",
            "`low_light` transform. These are not corrected COCO benchmark scores.", "",
        ])
    else:
        lines.extend([
            "Corrected evaluation with a matched-brightness control. This is one",
            "seed; confidence intervals and repeated-seed analysis are not included.", "",
        ])
    lines.extend([
        "A shared display format does not make legacy and corrected experiments",
        "directly comparable. Correlation alone does not establish detection benefit.", "",
    ])
    for metric, heading in zip(METRICS, ("mAP at IoU 0.50:0.95", "mAP at IoU 0.50")):
        lines.extend([
            f"## {heading}", "",
            "| Condition | Baseline | AD only | AD + LoG | AD + brightness control |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for condition in CONDITIONS:
            methods = summary["map_results"][condition]
            values = [format_value(methods[method][metric]) if method in methods else "Not run"
                      for method in METHODS]
            lines.append(f"| `{condition}` | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend([
        "## LoG/Gradient Correlation", "",
        "| Condition | Mean | Standard deviation |",
        "| --- | ---: | ---: |",
    ])
    for condition in CONDITIONS:
        correlation = summary["corr_summary"][condition]
        lines.append(f"| `{condition}` | {format_value(correlation['mean'])} | "
                     f"{format_value(correlation['std'])} |")
    lines.extend([
        "", "## Common Parameters", "",
        "| Parameter | Recorded value |", "| --- | ---: |",
    ])
    for key in PARAMETERS:
        lines.append(f"| `{key}` | {summary['params'][key]} |")
    lines.extend([
        "", "These are the parameters shared by the compact result schema, not the",
        "complete inference/degradation configuration. See the protocol and source",
        "artifact for additional metadata and known provenance gaps.", "",
    ])
    return "\n".join(lines)


def build_outputs(run_dir, root=ROOT):
    root = Path(root).resolve()
    experiment_dir = root / "experiments" / CURRENT_EXPERIMENT
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != experiment_dir / "runs":
        raise ValueError("Select a run directly under experiments/02/runs")
    data = read_json(run_dir / "results.json")
    manifest = read_json(run_dir / "manifest.json")
    status = read_json(run_dir / "status.json")
    if (status.get("status") != "complete" or data.get("run_kind") != "full"
            or manifest.get("run_kind") != "full"
            or manifest.get("experiment") not in CURRENT_MANIFEST_IDS):
        raise ValueError("Only a completed full experiment may provide the public summary")
    dataset = manifest["dataset"]
    count = data["image_count"]
    ids = [record["id"] for record in dataset["images"]]
    if (dataset.get("all_images") is not True or count != dataset["expected_images"]
            or count != dataset["evaluated_images"] or len(ids) != count
            or len(set(ids)) != count):
        raise ValueError("Full-run image counts or IDs are inconsistent")
    if (len(manifest["conditions"]) != len(CONDITIONS)
            or set(manifest["conditions"]) != set(CONDITIONS)
            or len(manifest["methods"]) != len(METHODS)
            or set(manifest["methods"]) != set(METHODS)
            or data["inference_calls"] != count * len(CONDITIONS) * len(METHODS)
            or status["inference_calls"] != data["inference_calls"]):
        raise ValueError("Full-run conditions, methods or inference count are inconsistent")
    current = compact_summary(data, manifest)
    if any(set(methods) != set(METHODS) for methods in current["map_results"].values()):
        raise ValueError("The corrected experiment requires all four methods")
    outputs = {
        run_dir / "summary.json": json.dumps(current, indent=2, allow_nan=False) + "\n",
        run_dir / "RESULTS.md": render_report(
            f"{CURRENT_EXPERIMENT}/runs/{run_dir.name}", current, "results.json", "../../README.md"),
    }
    for name in LEGACY_RUNS:
        directory = root / "experiments" / LEGACY_EXPERIMENT / "runs" / name
        legacy = compact_summary(read_json(directory / "results.json"))
        outputs[directory / "summary.json"] = json.dumps(legacy, indent=2, allow_nan=False) + "\n"
        outputs[directory / "RESULTS.md"] = render_report(
            f"{LEGACY_EXPERIMENT}/runs/{name}", legacy, "results.json")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Completed corrected full run to format alongside the two legacy runs")
    parser.add_argument("--check", action="store_true", help="Check generated files without writing")
    arguments = parser.parse_args(argv)
    outputs = build_outputs(arguments.run_dir)
    stale = [path for path, content in outputs.items()
             if not path.is_file() or path.read_text(encoding="utf-8") != content]
    if arguments.check:
        if stale:
            parser.exit(1, "Missing or outdated summaries: " + ", ".join(
                str(path.relative_to(ROOT)) for path in stale) + "\n")
        print("All three result summaries use the current common format.")
        return
    for path in stale:
        if path.is_symlink():
            raise ValueError(f"Refusing to replace a symlink: {path.relative_to(ROOT)}")
    for path in stale:
        path.write_text(outputs[path], encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
