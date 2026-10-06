"""Check experiment packaging without importing the ML dependencies."""

import ast
from contextlib import chdir
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "001_pilot_5": 5,
    "002_full_5000": 5000,
}
# JSON files are original; the log fingerprint includes the documented privacy edit.
ARCHIVE_HASHES = {
    "001_pilot_5/results.json":
        "8265a90458cca09c989245e2ec0610aa1f4e674e8e0bef37eb900cf0759fbe8e",
    "002_full_5000/results.json":
        "287c188061661b1d7702895b434fc18aa095496fb8ea5c8382a5c981c1802bb4",
    "002_full_5000/run.log":
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


def isolated_main(path, results_dir, model_path):
    main = next(node for node in script_tree(path).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    namespace = {
        "os": os,
        "RESULTS_DIR": str(results_dir),
        "MODEL_PATH": str(model_path),
    }
    module = ast.Module(body=[main], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["main"]


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
        paths.extend((ROOT / "experiments" / "003_corrected_full").glob("*.py"))
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
                                     ROOT / "experiments" / name)
                    self.assertEqual(Path(config["COCO_IMG_DIR"]),
                                     ROOT / "shared/coco/images/val2017")
                    self.assertEqual(Path(config["COCO_ANN_FILE"]),
                                     ROOT / "shared/coco/annotations"
                                     / "instances_val2017.json")
                    self.assertEqual(Path(config["MODEL_PATH"]),
                                     ROOT / "shared/yolo11n.pt")
                    self.assertEqual(config["N_IMAGES"], image_count)

    def test_archived_parameters_match_their_code(self):
        for name in ("001_pilot_5", "002_full_5000"):
            with self.subTest(experiment=name):
                path = script_path(name)
                result = json.loads(path.with_name("results.json").read_text(
                    encoding="utf-8"))
                config = configuration(path)
                for key, value in result["params"].items():
                    self.assertEqual(config[key], value, key)

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
                    str(script_path("003_corrected_full")), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
