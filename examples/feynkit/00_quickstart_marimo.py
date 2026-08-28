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
    # FeynKit one-loop quickstart

    This ten-minute tour goes from a normalized quantum-field-theory model to
    typed Feynman diagrams. FeynKit is part of the same `symbolica`
    installation and shares Symbolica's expression engine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    Install the community distribution with `pip install symbolica`. We use
    the conventional short alias `fk` so the physics namespace stays visible.

    The data path below is resolved relative to this app, so it works whether
    Marimo is launched from the repository root or this examples directory.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import symbolica.community.feynkit as fk

    data_dir = Path(__file__).resolve().parent / "data"
    return data_dir, fk


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load a normalized model

    A normalized JSON model is portable and does not require Python UFO
    tooling. The bundled scalar model is deliberately small enough for an
    interactive tutorial.
    """)
    return


@app.cell
def _(data_dir, fk, mo, table):
    model = fk.Model(data_dir / "scalars_2p_3p.json")

    table(
        [
            {
                "model": model.name,
                "particles": len(model.particles),
                "parameters": len(model.parameters),
                "vertices": len(model.vertex_rules),
            }
        ]
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Describe and generate an amplitude

    Particle selectors may be names such as `"scalar_0"`, PDG codes, or
    explicit `ParticleSelector` objects. Here `loops=1` requests exactly one
    loop. Enabling self-loops lets the generator enumerate the complete set of
    allowed one-loop topologies; below, we select the first diagram without a
    self-edge for a particularly clear visualization.
    """)
    return


@app.cell
def _(fk, mo, model, table):
    _options = fk.GenerationOptions(max_vertices=3, allow_self_loops=True)
    _options.add_vertex_allow(["V_3_SCALAR_000"])

    generated = model.generate_diagrams(
        incoming=["scalar_0"],
        outgoing=["scalar_0", "scalar_0"],
        loops=1,
        options=_options,
    )

    table(
        [
            {
                "diagrams": len(generated),
                "topologies considered": generated.report.topology_count,
                "interaction assignments": (
                    generated.report.interaction_assignment_count
                ),
            }
        ]
    )
    return (generated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect a typed diagram

    Vertices and edges retain physics metadata. Methods ending in
    `_expression` return native `symbolica.core.Expression` objects, ready for
    symbolic manipulation. Both types implement notebook rich display; Marimo
    table cells use `mo.as_html(...)` to preserve it.
    """)
    return


@app.cell
def _(generated, mo, model, table):
    diagram = next(
        item
        for item in generated.diagrams
        if all(edge.source != edge.target for edge in item.edges)
    )
    diagram.validate(model)
    _factor = diagram.overall_factor_expression()

    table(
        [
            {
                "diagram": mo.as_html(diagram),
                "loops": diagram.loop_count,
                "vertices": len(diagram.vertices),
                "edges": len(diagram.edges),
                "overall factor": mo.as_html(_factor.formatted()),
            }
        ],
        column_widths={"diagram": 440, "overall factor": 180},
    )
    return (diagram,)


@app.cell
def _(diagram):
    diagram
    return


@app.cell
def _(diagram, mo, table):
    _edge_rows = [
        {
            "id": edge.id,
            "from": edge.source,
            "to": edge.target,
            "particle": edge.particle_name,
            "pdg": edge.particle_pdg,
        }
        for edge in diagram.edges
    ]
    table(_edge_rows)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    - **01 — Models and diagrams** covers particle/parameter lookup, parameter
      cards, loop ranges, graph round trips, and loop-momentum bases.
    - **02 — CFF and Symbolica** constructs a Cross-Free Family representation
      and converts it to a Symbolica expression.
    - **03 — Kinematics and jets** covers the mostly-minus metric, boosts,
      rotations, angular distances, and generalized-​\(k_T\) clustering.
    """)
    return


if __name__ == "__main__":
    app.run()
