"""Load the bundled raw UFO fixture through FeynKit's public Python API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MINIMUM_PYTHON = (3, 11)
UFO_EXTRA = "symbolica[feynkit-ufo]"
UFO_PACKAGE = "ufo-model-loader"
INTERACTION_ENV = "UFO_SCALARS_MODEL_N_POINT_INTERACTIONS"
EXPECTED_DIAGNOSTICS = {
    "order_count": 2,
    "model_parameter_count": 6,
    "particle_count": 3,
    "propagator_count": 3,
    "lorentz_structure_count": 2,
    "coupling_count": 1,
    "vertex_rule_count": 16,
    "function_count": 7,
    "form_factor_count": 0,
    "parameter_value_count": 3,
}
NOTEBOOK_PACKAGES = ("ipykernel", "nbclient", "nbformat")


def _dependency_version() -> str:
    if sys.version_info < MINIMUM_PYTHON:
        found = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(
            f"Raw UFO loading requires Python 3.11 or newer; found {found}."
        )
    try:
        return version(UFO_PACKAGE)
    except PackageNotFoundError as error:
        raise SystemExit(
            f"Raw UFO loading is optional. Install it with: pip install '{UFO_EXTRA}'"
        ) from error


def _check_notebook(notebook: Path, runner: Path) -> None:
    missing = []
    for package in NOTEBOOK_PACKAGES:
        try:
            version(package)
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Executing the UFO tutorial also needs the notebook test packages. "
            f"Install them with: pip install {packages}"
        )

    # A fresh process gives the notebook the same isolation as the general
    # tutorial runner and lets it test its imports from a clean interpreter.
    subprocess.run(
        [sys.executable, str(runner), "--one", str(notebook)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="run the raw-loader assertions without executing its tutorial notebook",
    )
    args = parser.parse_args()

    loader_version = _dependency_version()

    root = Path(__file__).resolve().parents[1]
    notebook = root / "examples" / "feynkit" / "04_ufo_loading.ipynb"
    if not args.api_only:
        _check_notebook(notebook, root / "scripts" / "run_feynkit_notebooks.py")

    # Import only after the optional-dependency check so a missing extra gets a
    # focused, actionable message. The notebook process must also finish before
    # this import so restricted single-instance Symbolica installations work.
    from symbolica.community.feynkit import UfoLoader

    source = (root / "examples" / "feynkit" / "data" / "ufo_scalars").resolve()

    previous_interactions = os.environ.get(INTERACTION_ENV)
    os.environ[INTERACTION_ENV] = "2,3"
    try:
        loaded = UfoLoader(restriction_name="default").load(source)
    finally:
        if previous_interactions is None:
            os.environ.pop(INTERACTION_ENV, None)
        else:
            os.environ[INTERACTION_ENV] = previous_interactions

    diagnostics = loaded.diagnostics
    actual_diagnostics = {
        name: getattr(diagnostics, name) for name in EXPECTED_DIAGNOSTICS
    }
    assert Path(diagnostics.source).resolve() == source
    assert diagnostics.restriction_name == "default"
    assert diagnostics.simplify_model is True
    assert diagnostics.wrap_indices_in_lorentz_structures is True
    assert actual_diagnostics == EXPECTED_DIAGNOSTICS

    model = loaded.model
    assert model.name == "ufo_scalars"
    assert model.restriction == "default"
    assert [(particle.name, particle.pdg_code) for particle in model.particles] == [
        ("scalar_0", 1000),
        ("scalar_1", 1001),
        ("scalar_2", 1002),
    ]

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    counts = ", ".join(
        f"{name.removesuffix('_count')}={value}"
        for name, value in actual_diagnostics.items()
    )
    print(
        f"UFO smoke check passed with Python {python_version} and "
        f"{UFO_PACKAGE} {loader_version}."
    )
    if not args.api_only:
        print(f"Executed {notebook.relative_to(root)} in a fresh kernel.")
    print(f"Loaded {model.name}-{model.restriction}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
