# FeynKit tutorials

These focused tutorials introduce the public particle-physics API in a useful
working order. Every tutorial is available both as a Jupyter notebook
(`.ipynb`) and as a native reactive Marimo app (`*_marimo.py`):

1. [`00_quickstart.ipynb`](00_quickstart.ipynb) /
   [`00_quickstart_marimo.py`](00_quickstart_marimo.py) — normalized model to
   a rendered one-loop Feynman diagram.
2. [`01_models_and_diagrams.ipynb`](01_models_and_diagrams.ipynb) /
   [`01_models_and_diagrams_marimo.py`](01_models_and_diagrams_marimo.py) —
   particles, parameter cards, loop ranges, graph interchange, and
   loop-momentum bases.
3. [`02_cff_and_symbolica.ipynb`](02_cff_and_symbolica.ipynb) /
   [`02_cff_and_symbolica_marimo.py`](02_cff_and_symbolica_marimo.py) —
   Cross-Free Family surfaces, orientations, denominator products, and native
   Symbolica expressions.
4. [`03_kinematics_and_jets.ipynb`](03_kinematics_and_jets.ipynb) /
   [`03_kinematics_and_jets_marimo.py`](03_kinematics_and_jets_marimo.py) —
   metric conventions, on-shell momenta, boosts, rotations, angular distances,
   and generalized-kT jets.
5. [`04_ufo_loading.ipynb`](04_ufo_loading.ipynb) /
   [`04_ufo_loading_marimo.py`](04_ufo_loading_marimo.py) — optional raw UFO
   loading, restriction cards, normalization diagnostics, and reuse in diagram
   generation.
6. [`05_one_loop_local_point.ipynb`](05_one_loop_local_point.ipynb) /
   [`05_one_loop_local_point_marimo.py`](05_one_loop_local_point_marimo.py) —
   one-loop momentum routing, native Feynman-rule expressions, CFF conversion,
   and denominator inspection at two routed loop-momentum points.
7. [`06_one_loop_qcd_numerators.ipynb`](06_one_loop_qcd_numerators.ipynb) /
   [`06_one_loop_qcd_numerators_marimo.py`](06_one_loop_qcd_numerators_marimo.py)
   — gluon, ghost, and bottom-quark contributions to the one-loop gluon
   self-energy, with reactively selected native Symbolica numerators and
   spenso/idenso contractions of their Lorentz, Dirac, and SU(3) indices.
8. [`07_tensor_reduction.ipynb`](07_tensor_reduction.ipynb) /
   [`07_tensor_reduction_marimo.py`](07_tensor_reduction_marimo.py) —
   symmetry-aware Lorentz tensor reduction for vacuum graphs, from the basic
   rank-two projector through compact rank-six and rank-twenty examples.

## Marimo

The repository venv contains Marimo and the `ty` language server. Start the
remote-capable editor from the repository root:

```bash
source .venv-feynkit/bin/activate
export SYMBOLICA_LICENSE='<your Symbolica license>'
scripts/start_feynkit_marimo.sh
```

The launcher listens on `0.0.0.0:2718`, runs headlessly, and uses the fixed
access URL [http://127.0.0.1:2718/?access_token=6jr3QPRRx8gxWnt1c5X2Jg](http://127.0.0.1:2718/?access_token=6jr3QPRRx8gxWnt1c5X2Jg).
The token is passed through standard input rather than exposed in the Marimo
process arguments. Override the bind address or port with `MARIMO_HOST` and
`MARIMO_PORT`. Set `MARIMO_NOTEBOOK_PATH` to open one app directly.

Use an SSH tunnel (`MARIMO_HOST=127.0.0.1`) or an HTTPS reverse proxy on an
untrusted network, because a plain HTTP connection does not encrypt the token.
For SSH, start Marimo on the remote host with:

```bash
MARIMO_HOST=127.0.0.1 scripts/start_feynkit_marimo.sh
```

Then forward the port from your local machine, replacing `user@remote-host`:

```bash
ssh -N -L 2718:127.0.0.1:2718 user@remote-host
```

Open the token-bearing `http://127.0.0.1:2718/...` URL printed by Marimo in
your local browser.

Behind HTTPS, set `MARIMO_SESSION_COOKIE_SECURE=1` so the session cookie is
marked secure. Project configuration in `pyproject.toml` enables diagnostics,
uses `ty` as the sole Python LSP, disables Copilot, and selects `uv` for Marimo
package management. Marimo 0.24 uses `ty` for diagnostics; its editor retains
its built-in behavior for other language features.

To reproduce the editor dependencies in another venv:

```bash
uv pip install --python .venv-feynkit/bin/python marimo ty
```

Marimo's LSP bridge also needs Node.js. This workspace pins Node 22 inside the
venv; on another machine, install Node.js on `PATH` or create the same local
pin with:

```bash
nix build nixpkgs#nodejs_22 --out-link .venv-feynkit/nodejs
```

Validate all native apps without starting the editor:

```bash
marimo check --strict examples/feynkit/*_marimo.py
```

## Jupyter

Install the project, start Jupyter in this directory, and select a Python 3
kernel containing `symbolica`:

```bash
pip install symbolica jupyter marimo
jupyter lab
```

Both tutorial formats are deterministic, make their conventions explicit, and
surface the key physics results directly for inspection. The bundled model in
`data/` includes both the small scalar teaching fixture and FeynKit's
authoritative normalized Standard Model fixture.

## Optional: load a raw UFO model

The notebooks use normalized JSON so they stay portable. To exercise the raw
UFO import boundary, use Python 3.11 or newer and install the optional loader:

```bash
pip install 'symbolica[feynkit-ufo]' jupyter
python scripts/check_feynkit_ufo.py
```

Run the check from the repository root. It executes `04_ufo_loading.ipynb` in a
fresh kernel, then independently loads `data/ufo_scalars` through
`UfoLoader.load`, applies `restrict_default.dat`, and verifies the
normalized model and loader diagnostics. It deliberately selects only the
two- and three-point interactions so the result is quick and deterministic.

The general `python scripts/run_feynkit_notebooks.py` command reports this
optional tutorial as skipped, keeping the seven core tutorials compatible with
Python 3.10 and a base `symbolica` install. After installing the UFO extra on
Python 3.11 or newer, use `--include-optional` to execute all eight. Use
`python scripts/check_feynkit_ufo.py --api-only` when notebook packages are not
installed and only the loader smoke check is needed.
