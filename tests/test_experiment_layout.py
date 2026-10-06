"""Check experiment packaging without importing the ML dependencies."""

import ast
import argparse
from contextlib import chdir
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "01": 5000,
}
# JSON files are original; the log fingerprint includes the documented privacy edit.
ARCHIVE_HASHES = {
    "01/runs/pilot_5/results.json":
        "8265a90458cca09c989245e2ec0610aa1f4e674e8e0bef37eb900cf0759fbe8e",
    "01/runs/full_5000/results.json":
        "287c188061661b1d7702895b434fc18aa095496fb8ea5c8382a5c981c1802bb4",
    "01/runs/full_5000/run.log":
        "c687653de340cd97be7ed75c3cb050c91559bf2896ffb1297b7193af9666d384",
}
CONFIG_NAMES = {
    "EXPERIMENT_DIR", "PROJECT_ROOT", "COCO_IMG_DIR", "COCO_ANN_FILE",
    "MODEL_PATH", "RESULTS_DIR", "N_IMAGES", "RANDOM_SEED", "AD_KAPPA",
    "AD_NITER", "AD_GAMMA", "LOG_SIGMA", "LOG_BETA",
}


def script_path(name):
    return ROOT / "experiments" / name / "run_experiment.py"


def script_tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def configuration(path):
    # Execute only path/parameter assignments, never imports or inference.
    assignments = [
        node for node in script_tree(path).body
        if isinstance(node, ast.Assign)
        and all(isinstance(target, ast.Name) and target.id in CONFIG_NAMES
                for target in node.targets)
    ]
    namespace = {"os": os, "__file__": str(path)}
    module = ast.Module(body=assignments, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def isolated_main(path, results_dir, model_path, extra_namespace=None):
    main = next(node for node in script_tree(path).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    namespace = dict(extra_namespace or {})
    namespace.update({
        "os": os,
        "RESULTS_DIR": str(results_dir),
        "MODEL_PATH": str(model_path),
    })
    module = ast.Module(body=[main], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["main"]


def isolated_run_configuration(experiment_dir):
    path = script_path("01")
    function = next(node for node in script_tree(path).body
                    if isinstance(node, ast.FunctionDef) and node.name == "configure_run")
    namespace = {"os": os, "argparse": argparse, "datetime": datetime,
                 "timezone": timezone, "re": re, "uuid": uuid,
                 "EXPERIMENT_DIR": str(experiment_dir), "N_IMAGES": 5000}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ExperimentLayoutTests(unittest.TestCase):
    def test_notebook_code_cells_compile_without_execution(self):
        path = ROOT / "docs" / "AD_LoG_Experiment_Ubuntu.ipynb"
        if not path.exists():
            self.skipTest("Optional local notebook is not included in the repository")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"]
                      if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        for index, cell in enumerate(code_cells):
            with self.subTest(cell=cell.get("id", index)):
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                compile(source, f"{path}:code-cell-{index}", "exec")

    def test_scripts_compile_without_importing_dependencies(self):
        paths = [script_path(name) for name in EXPERIMENTS]
        paths.extend((ROOT / "experiments" / "02").glob("*.py"))
        paths.append(ROOT / "run_experiment.py")
        for path in paths:
            with self.subTest(path=path):
                compile(script_tree(path), str(path), "exec")

    def test_configuration_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            for name, image_count in EXPERIMENTS.items():
                with self.subTest(experiment=name):
                    config = configuration(script_path(name))
                    self.assertEqual(Path(config["PROJECT_ROOT"]), ROOT)
                    self.assertEqual(Path(config["EXPERIMENT_DIR"]),
                                     ROOT / "experiments" / name)
                    self.assertEqual(Path(config["RESULTS_DIR"]),
                                     ROOT / "experiments" / name / "runs")
                    self.assertEqual(Path(config["COCO_IMG_DIR"]),
                                     ROOT / "shared/coco/images/val2017")
                    self.assertEqual(Path(config["COCO_ANN_FILE"]),
                                     ROOT / "shared/coco/annotations"
                                     / "instances_val2017.json")
                    self.assertEqual(Path(config["MODEL_PATH"]),
                                     ROOT / "shared/yolo11n.pt")
                    self.assertEqual(config["N_IMAGES"], image_count)

    def test_archived_parameters_match_their_code(self):
        for name in ("pilot_5", "full_5000"):
            with self.subTest(run=name):
                path = script_path("01")
                run_dir = path.parent / "runs" / name
                result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
                recorded_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(recorded_config["params"], result["params"])
                config = configuration(path)
                for key, value in result["params"].items():
                    if key != "N_IMAGES":
                        self.assertEqual(config[key], value, key)
                selected = isolated_run_configuration(path.parent)
                selected["configure_run"](["--images", str(result["params"]["N_IMAGES"]),
                                            "--run-name", f"new_{name}"])
                self.assertEqual(selected["N_IMAGES"], result["params"]["N_IMAGES"])

    def test_runs_share_one_script_and_select_separate_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for count in (5, 5000):
                selected = isolated_run_configuration(directory)
                selected["configure_run"](["--images", str(count), "--run-name", f"new_{count}"])
                self.assertEqual(selected["N_IMAGES"], count)
                self.assertEqual(Path(selected["RESULTS_DIR"]), directory / "runs" / f"new_{count}")
                self.assertFalse(Path(selected["RESULTS_DIR"]).exists())
            for name in ("pilot_5", "full_5000"):
                self.assertFalse((script_path("01").parent / "runs" / name
                                  / "run_experiment.py").exists())

    def test_run_selection_rejects_existing_directories_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "runs" / "archive").mkdir(parents=True)
            selected = isolated_run_configuration(directory)
            with self.assertRaises(FileExistsError):
                selected["configure_run"](["--run-name", "archive"])
            for name in ("../escape", "/absolute", "a/b", "..", "name with spaces"):
                with self.subTest(name=name), patch("sys.stderr"), self.assertRaises(SystemExit):
                    selected["configure_run"](["--run-name", name])
            for count in (0, -1, 5001):
                with self.subTest(count=count), patch("sys.stderr"), self.assertRaises(SystemExit):
                    selected["configure_run"](["--images", str(count)])

    def test_default_run_names_are_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = isolated_run_configuration(temporary)
            second = isolated_run_configuration(temporary)
            first["configure_run"]([])
            second["configure_run"]([])
            self.assertEqual(first["N_IMAGES"], 5000)
            self.assertNotEqual(first["RESULTS_DIR"], second["RESULTS_DIR"])

    def test_new_run_records_selected_parameters_and_code_before_processing(self):
        path = script_path("01")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weights = directory / "weights.pt"
            weights.write_bytes(b"fixture")
            selected = isolated_run_configuration(directory)
            selected["configure_run"](["--images", "5", "--run-name", "new_pilot"])
            run_dir = Path(selected["RESULTS_DIR"])
            dependencies = configuration(path)
            dependencies.update(N_IMAGES=selected["N_IMAGES"], json=json,
                                shutil=shutil, __file__=str(path))
            numpy = Mock()
            numpy.random.seed.side_effect = RuntimeError("stop before image processing")
            dependencies["np"] = numpy
            main = isolated_main(path, run_dir, weights, dependencies)
            with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "stop before image processing"):
                main()
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["experiment"], "01")
            self.assertEqual(config["params"]["N_IMAGES"], 5)
            self.assertEqual(config["params"]["RANDOM_SEED"], 42)
            self.assertEqual((run_dir / "source_snapshot/run_experiment.py").read_bytes(), path.read_bytes())
            self.assertFalse((run_dir / "results.json").exists())

    def test_archived_artifacts_match_documented_fingerprints(self):
        for relative_path, expected_hash in ARCHIVE_HASHES.items():
            with self.subTest(artifact=relative_path):
                path = ROOT / "experiments" / relative_path
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 expected_hash)

    def test_existing_result_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = directory / "results.json"
            result.write_text("archived result", encoding="utf-8")
            for name in EXPERIMENTS:
                with self.subTest(experiment=name):
                    main = isolated_main(script_path(name), directory,
                                         directory / "missing.pt")
                    with self.assertRaises(FileExistsError):
                        main()
                    self.assertEqual(result.read_text(encoding="utf-8"),
                                     "archived result")

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlinks are unavailable")
    def test_dangling_result_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = directory / "results.json"
            result.symlink_to(directory / "missing-result.json")
            for name in EXPERIMENTS:
                with self.subTest(experiment=name):
                    main = isolated_main(script_path(name), directory,
                                         directory / "missing.pt")
                    with self.assertRaises(FileExistsError):
                        main()
                    self.assertTrue(result.is_symlink())

    def test_missing_weights_stop_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in EXPERIMENTS:
                with self.subTest(experiment=name):
                    main = isolated_main(script_path(name), directory,
                                         directory / "missing.pt")
                    with self.assertRaises(FileNotFoundError):
                        main()
                    self.assertFalse((directory / "results.json").exists())

    def test_result_files_use_exclusive_creation(self):
        for name in EXPERIMENTS:
            with self.subTest(experiment=name):
                writes = [
                    node for node in ast.walk(script_tree(script_path(name)))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "open"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "out_path"
                ]
                self.assertEqual(len(writes), 1)
                self.assertEqual(ast.literal_eval(writes[0].args[1]), "x")

    def test_root_launcher_dispatches_from_any_directory(self):
        launcher = ROOT / "run_experiment.py"
        namespace = {"__name__": "__main__", "__file__": str(launcher)}
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            with patch("runpy.run_path") as run_path:
                exec(compile(script_tree(launcher), str(launcher), "exec"),
                     namespace)
                run_path.assert_called_once_with(
                    str(script_path("02")), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
