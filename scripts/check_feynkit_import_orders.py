"""Verify that FeynKit and Spenso share the installed Symbolica core."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "examples" / "feynkit" / "data" / "scalars_2p_3p.json"
IMPORT_ORDERS = (
    ("symbolica.community.feynkit", "symbolica.community.spenso"),
    ("symbolica.community.spenso", "symbolica.community.feynkit"),
)


def check_order(first: str, second: str) -> None:
    """Run one import order in a pristine interpreter."""

    script = f"""
import importlib
import sys

importlib.import_module({first!r})
importlib.import_module({second!r})
feynkit = importlib.import_module("symbolica.community.feynkit")
spenso = importlib.import_module("symbolica.community.spenso")
from symbolica import Expression

assert "symbolica.community.feynkit_native" in sys.modules
assert "_gammaloop" not in sys.modules
for exported in (
    feynkit.Model,
    feynkit.Generator,
    feynkit.FeynmanDiagram,
    feynkit.CffGenerator,
    feynkit.FourMomentum,
    feynkit.JetDefinition,
    feynkit.UfoLoader,
):
    assert exported.__module__ == "symbolica.community.feynkit"
for removed in (
    "ParticleLike",
    "LoopOrder",
    "load_ufo_model",
    "generate_diagrams",
    "build_cff",
    "cluster_jets",
):
    assert not hasattr(feynkit, removed)

model = feynkit.Model.from_path({str(MODEL_PATH)!r})
options = feynkit.GenerationOptions(max_vertices=3)
options.add_vertex_allow(["V_3_SCALAR_000"])
generated = model.generate_diagrams(
    ["scalar_0"],
    ["scalar_0", "scalar_0"],
    options=options,
)
factor = generated.diagrams[0].overall_factor_expression()
tensor = spenso.TensorName("T").to_expression()
assert type(factor) is Expression
assert type(factor + tensor) is Expression
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def main() -> int:
    for first, second in IMPORT_ORDERS:
        check_order(first, second)
        print(f"checked {first} -> {second}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
