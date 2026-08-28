import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial

    import marimo as mo

    table = partial(mo.ui.table, selection=None)
    return mo, table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Models, parameters, and diagram generation

    This tutorial treats a model as physics data: inspect particles and
    parameters, make an immutable parameter update, generate tree and loop
    diagrams, and inspect their graph structure.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import symbolica.community.feynkit as fk

    _data_file = Path(__file__).resolve().parent / "data" / "scalars_2p_3p.json"
    model = fk.Model(_data_file)
    return fk, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Particle content

    Lookups are available by model name or PDG code. Spin follows the UFO
    convention \(2s+1\), so the scalar entries have `spin == 1`.
    """)
    return


@app.cell
def _(mo, model, table):
    _particle_rows = [
        {
            "name": particle.name,
            "pdg": particle.pdg_code,
            "spin (2s+1)": particle.spin,
            "color rep": particle.color,
            "mass parameter": particle.mass_parameter,
            "massless": particle.is_massless,
        }
        for particle in model.particles
    ]

    table(_particle_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter cards and immutability

    `ParameterCard` is mutable configuration, while
    `Model.with_parameter_card` returns a new model. Without an evaluator,
    changing an external parameter intentionally invalidates dependent
    internal parameters and couplings rather than leaving stale values.
    """)
    return


@app.cell
def _(fk, mo, model, table):
    _card = model.default_parameter_card()
    _card.set("lam", 2.5)
    _updated = model.with_parameter_card(_card)

    try:
        _dependent_coupling = _updated.coupling("SCALAR_COUPLING").value
    except fk.ModelError:
        _dependent_coupling = "not evaluated after the parameter update"

    table(
        [
            {
                "original lambda": model.parameter("lam").value,
                "updated lambda": _updated.parameter("lam").value,
                "dependent coupling after invalidation": _dependent_coupling,
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Processes and inclusive loop ranges

    `Process.amplitude` and `Process.cross_section` make the requested graph
    semantics explicit. Cross-section generation constructs cross-section
    graph structures; it does not numerically integrate phase space. Loop
    bounds are inclusive.
    """)
    return


@app.cell
def _(fk, mo, table):
    _process = fk.Process.amplitude(
        ["scalar_0"], [1000, "scalar_0"]
    ).with_loop_count(0, 1)

    table(
        [
            {
                "kind": str(_process.generation_type),
                "incoming": ", ".join(str(item) for item in _process.incoming),
                "outgoing": ", ".join(
                    str(item) for item in _process.outgoing_alternatives[0]
                ),
                "loops": str(_process.loop_count),
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `GenerationOptions` is a mutable configuration object. Methods named
    `add_*` and the current `set_*_filter` methods add filters; configure each
    filter family once to avoid duplicate-filter errors.
    """)
    return


@app.cell
def _(fk, mo, model, table):
    _options = fk.GenerationOptions(max_vertices=3, allow_self_loops=True)
    _options.add_vertex_allow(["V_3_SCALAR_000"])

    generated = model.generate_diagrams(
        incoming=["scalar_0"],
        outgoing=[1000, "scalar_0"],
        loops=(0, 1),
        options=_options,
    )

    table(
        [
            {
                "retained": generated.report.retained_count,
                "loop orders": ", ".join(
                    str(order)
                    for order in sorted(
                        {diagram.loop_count for diagram in generated.diagrams}
                    )
                ),
            }
        ]
    )
    return (generated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graph interchange and validation

    Diagrams round-trip through JSON for lossless storage and through DOT for
    graph-tool interoperability. Validate imported diagrams against the model
    before using them downstream.
    """)
    return


@app.cell
def _(fk, generated, mo, model, table):
    _loop_diagram = next(
        diagram
        for diagram in generated.diagrams
        if diagram.loop_count == 1
        and all(edge.source != edge.target for edge in diagram.edges)
    )

    from_json = fk.FeynmanDiagram.from_json(_loop_diagram.to_json())
    _from_dot = fk.FeynmanDiagram.from_dot(_loop_diagram.to_dot())
    from_json.validate(model)
    _from_dot.validate(model)

    table(
        [
            {
                "diagram": mo.as_html(from_json),
                "loops": from_json.loop_count,
                "vertices": len(from_json.vertices),
                "edges": len(from_json.edges),
            }
        ],
        column_widths={"diagram": 440},
    )
    return (from_json,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loop-momentum bases

    A basis identifies loop edges, tree edges, dependent external momenta, and
    the signed loop/external momentum signature carried by every edge.
    """)
    return


@app.cell
def _(from_json, mo, table):
    _bases = from_json.loop_momentum_bases(limit=8)
    _basis = _bases[0]

    _momentum_rows = [
        {"edge": edge, "momentum": signature.format_momentum()}
        for edge, signature in _basis.edge_signatures.items()
    ]

    mo.vstack(
        [
            table(
                [
                    {
                        "number of bases returned": len(_bases),
                        "loop edges": _basis.loop_edges,
                        "tree edges": _basis.tree_edges,
                        "external edges": _basis.external_edges,
                    }
                ]
            ),
            mo.md("**Momentum carried by each edge**"),
            table(_momentum_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading UFO models

    For a raw UFO model directory, install the optional dependency on Python
    3.11 or newer:

    ```bash
    pip install "symbolica[feynkit-ufo]"
    ```

    Then configure an `fk.UfoLoader`, for example
    `fk.UfoLoader(restriction_name="massless").load(path)`. It returns a
    `LoadedModel` containing the normalized `model`, its `parameters`, and
    detailed loader `diagnostics`. Normalized JSON remains the reproducible,
    dependency-free choice for saved analyses.
    """)
    return

if __name__ == "__main__":
    app.run()
