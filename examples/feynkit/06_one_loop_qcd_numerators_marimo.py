import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from functools import partial

    import marimo as mo

    table = partial(mo.ui.table, selection=None)
    return mo, table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QCD: gluon self-energy numerators

    Generate the three non-scaleless one-loop QCD contributions to the gluon
    two-point function: a gluon loop, a Faddeev–Popov ghost loop, and a
    bottom-quark loop. Then choose a graph interactively and inspect its
    instantiated Feynman rules as native Symbolica expressions.

    A full Standard-Model generation would produce one quark loop per flavor.
    Here we retain only `b`/`b~` at generation time, so the three displayed
    graphs correspond directly to the three familiar loop-field classes.
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
    ## Load FeynKit's normalized Standard Model

    The bundled `sm.json` is copied byte-for-byte from FeynKit's authoritative
    normalized model fixture. Loading JSON keeps this example deterministic and
    avoids the optional Python UFO loader. We use only its QCD sector below.
    """)
    return


@app.cell
def _(data_dir, fk, table):
    model = fk.Model.from_path(data_dir / "sm.json")
    gluon = model.particle("g")

    table(
        [
            {
                "model": model.name,
                "particles": len(model.particles),
                "interaction rules": len(model.vertex_rules),
                "gluon PDG": gluon.pdg_code,
                "gluon massless": gluon.is_massless,
            }
        ],
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the one-loop QCD gluon self-energy

    `loops=1` fixes the loop order, while the coupling-order bounds select
    exactly \(g_s^2\) and exclude electroweak insertions. `add_particle_veto`
    removes the other five quark flavors before graph generation; both signs
    of each PDG code are listed so the restriction covers particles and
    antiparticles. Self-loops stay disabled, excluding the massless
    four-gluon tadpole, which is scaleless and vanishes in dimensional
    regularization.

    FeynKit automatically instantiates each vertex and propagator rule while
    building the diagrams.
    """)
    return


@app.cell
def _(fk, model, table):
    options = fk.GenerationOptions(max_vertices=2)
    options.set_coupling_orders({"QCD": (2, 2), "QED": (0, 0)})
    options.add_particle_veto([1, -1, 2, -2, 3, -3, 4, -4, 6, -6])

    generated = model.generate_diagrams(
        incoming=["g"],
        outgoing=["g"],
        loops=1,
        options=options,
    )

    table(
        [
            {
                "completed": generated.report.completed,
                "retained diagrams": len(generated),
                "topologies considered": generated.report.topology_count,
                "interaction assignments": (
                    generated.report.interaction_assignment_count
                ),
            }
        ]
    )
    return (generated,)


@app.cell
def _(fk):
    def internal_edges(
        diagram: fk.FeynmanDiagram,
    ) -> list[fk.DiagramEdge]:
        external_vertices = {
            vertex.id for vertex in diagram.vertices if vertex.is_external
        }
        return [
            edge
            for edge in diagram.edges
            if edge.source not in external_vertices
            and edge.target not in external_vertices
        ]

    return (internal_edges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three loop-field contributions

    Internal edge metadata identifies the loop field without parsing graph
    labels. The rows are ordered as gluon, ghost, and bottom-quark contributions
    even though generator names are assigned independently of presentation.
    Each `diagram` cell delegates to `FeynmanDiagram._repr_html_()`, producing a
    Linnest-rendered graph rather than a string snapshot.
    """)
    return


@app.cell
def _(generated, internal_edges, mo, table):
    graph_catalog = (
        ("gluon", "Gluon loop", 21, "g"),
        ("ghost", "Ghost loop", 9000005, "ghG / ghG~"),
        ("bottom", "Bottom-quark loop", 5, "b / b~"),
    )
    _diagram_by_pdg = {
        abs(internal_edges(_diagram)[0].particle_pdg): _diagram
        for _diagram in generated.diagrams
        if internal_edges(_diagram)
    }
    _expected_pdgs = {_pdg for _, _, _pdg, _ in graph_catalog}
    if len(generated) != 3 or set(_diagram_by_pdg) != _expected_pdgs:
        raise RuntimeError(
            "expected exactly the gluon, ghost, and bottom-quark bubbles"
        )

    diagrams_by_kind = {
        _kind: _diagram_by_pdg[_pdg]
        for _kind, _, _pdg, _ in graph_catalog
    }
    labels_by_kind = {
        _kind: _label for _kind, _label, _, _ in graph_catalog
    }

    table(
        [
            {
                "contribution": _label,
                "loop field": _field,
                "diagram": mo.as_html(diagrams_by_kind[_kind]),
            }
            for _kind, _label, _, _field in graph_catalog
        ],
        column_widths={"contribution": 190, "loop field": 130, "diagram": 560},
    )
    return diagrams_by_kind, graph_catalog, labels_by_kind


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose a graph

    The selector defaults to the gluon loop. Changing it invalidates the cells
    below, so the rendered graph and every Feynman-rule expression update
    reactively.
    """)
    return


@app.cell
def _(graph_catalog, mo):
    graph_selector = mo.ui.radio(
        options={_label: _kind for _kind, _label, _, _ in graph_catalog},
        value=graph_catalog[0][1],
        inline=True,
        label="**Loop contribution**",
    )
    graph_selector
    return (graph_selector,)


@app.cell
def _(diagrams_by_kind, graph_selector, labels_by_kind, mo):
    selected_kind = graph_selector.value
    selected_diagram = diagrams_by_kind[selected_kind]
    selected_label = labels_by_kind[selected_kind]

    mo.vstack(
        [
            mo.md(f"### {selected_label}"),
            mo.as_html(selected_diagram),
        ],
        align="center",
        gap=1,
    )
    return selected_diagram, selected_label


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Instantiated numerator

    The combined numerator is the native Symbolica product of the selected
    graph's interaction and internal-propagator factors. The diagram-wide
    factor contains its automorphism factor, external-fermion ordering sign,
    and the minus sign for each closed internal fermion loop.
    Multiplying the two gives the numerator with this combinatorial factor
    included.
    """)
    return


@app.cell
def _(mo, selected_diagram, selected_label, table):
    selected_numerator = selected_diagram.numerator_expression()
    selected_factor = selected_diagram.overall_factor_expression()
    weighted_numerator = selected_factor * selected_numerator

    table(
        [
            {
                "contribution": selected_label,
                "diagram factor": mo.as_html(selected_factor),
                "analytic numerator": mo.as_html(selected_numerator),
                "with diagram factor": mo.as_html(weighted_numerator),
            }
        ],
        column_widths={
            "contribution": 180,
            "diagram factor": 420,
            "analytic numerator": 650,
            "with diagram factor": 650,
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feynman-rule factors

    The combined expression can also be inspected rule by rule. Only internal
    vertices and propagators appear here; every factor remains a native
    Symbolica expression with rich mathematical rendering.
    """)
    return


@app.cell
def _(internal_edges, mo, selected_diagram, table):
    _vertex_rows = [
        {
            "vertex": vertex.id,
            "interaction rule": vertex.interaction,
            "analytic factor": mo.as_html(vertex.numerator_expression()),
        }
        for vertex in selected_diagram.vertices
        if not vertex.is_external
    ]
    _propagator_rows = [
        {
            "edge": edge.id,
            "loop field": edge.particle_name,
            "analytic factor": mo.as_html(edge.numerator_expression()),
        }
        for edge in internal_edges(selected_diagram)
    ]

    mo.ui.tabs(
        {
            "Interaction vertices": table(
                _vertex_rows,
                column_widths={"vertex": 80, "interaction rule": 160},
            ),
            "Internal propagators": table(
                _propagator_rows,
                column_widths={"edge": 80, "loop field": 130},
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading and reusing the result

    For the gluon graph, `G` is the strong coupling, `f(...)` is the SU(3)
    structure constant, and `Metric(...)` carries Lorentz contractions. The
    ghost numerator exposes its momentum insertion, while the bottom loop also
    carries Dirac and fundamental-color tensors. `Momentum(edge, index)` refers
    to the momentum entering an instantiated rule through the indicated graph
    edge. Sink/source indices make tensor contractions explicit.

    The returned values are ordinary `symbolica.Expression` objects, so they
    can be substituted, expanded, factored, differentiated, or passed directly
    into the rest of a Symbolica calculation. Multiplying by
    `selected_diagram.overall_factor_expression()` supplies the diagram-wide
    combinatorial factor when constructing an integrand numerator.
    """)
    return


@app.cell
def _():
    from symbolica import Expression, S
    from symbolica.community.idenso import (
        simplify_color,
        simplify_gamma,
        simplify_metrics,
        to_dots,
    )

    _a, _b, _c, _edge, _index_edge, _shift = S(
        "a_", "b_", "c_", "edge_", "index_edge_", "shift_"
    )
    _ufo_metric = S("UFO::Metric")
    _ufo_gamma = S("UFO::Gamma")
    _ufo_identity = S("UFO::Identity")
    _ufo_pslash = S("UFO::PSlash")
    _ufo_f = S("UFO::f")
    _ufo_t = S("UFO::T")

    _momentum = S("FeynKit::Momentum")
    _source_index = S("FeynKit::SourceIndex")
    _sink_index = S("FeynKit::SinkIndex")
    _slash_index = S("FeynKit::SlashIndex")
    _color_label = S("FeynKit::ColorLabel")
    _color_index = S("FeynKit::ColorIndex")

    _metric = S("spenso::g")
    _dot = S("spenso::dot")
    _gamma = S("spenso::gamma")
    _color_f = S("spenso::f")
    _color_t = S("spenso::t")
    _minkowski = S("spenso::mink")
    _bispinor = S("spenso::bis")
    _adjoint = S("spenso::coad")
    _fundamental = S("spenso::cof")
    _dual = S("spenso::dind")

    def contract_qcd_numerator(expression: Expression) -> Expression:
        """Contract a FeynKit QCD bubble numerator with spenso/idenso."""
        converted = expression.replace(
            _ufo_pslash(_a, _b, _momentum(_edge)),
            _gamma(
                _bispinor(4, _a),
                _bispinor(4, _b),
                _minkowski(4, _slash_index(_edge)),
            )
            * _momentum(_edge, _minkowski(4, _slash_index(_edge))),
        )
        converted = converted.replace(
            _ufo_gamma(_a, _b, _c),
            _gamma(
                _bispinor(4, _b),
                _bispinor(4, _c),
                _minkowski(4, _a),
            ),
        )
        converted = converted.replace(
            _ufo_identity(_a, _b),
            _metric(_bispinor(4, _a), _bispinor(4, _b)),
        )
        converted = converted.replace(
            _ufo_metric(_a, _b),
            _metric(_minkowski(4, _a), _minkowski(4, _b)),
        )
        converted = converted.replace(
            _ufo_f(_a, _b, _c),
            _color_f(
                _adjoint(8, _color_label(_a)),
                _adjoint(8, _color_label(_b)),
                _adjoint(8, _color_label(_c)),
            ),
        )
        converted = converted.replace(
            _ufo_t(_a, _b, _c),
            _color_t(
                _adjoint(8, _color_label(_a)),
                _fundamental(3, _color_label(_b)),
                _dual(_fundamental(3, _color_label(_c))),
            ),
        )
        converted = converted.replace(
            _color_label(_source_index(_index_edge, _shift)),
            _color_index(_index_edge, _shift),
        )
        converted = converted.replace(
            _color_label(_sink_index(_index_edge, _shift)),
            _color_index(_index_edge, _shift),
        )
        converted = converted.replace(
            _momentum(
                _edge, _source_index(_index_edge, _shift)
            ),
            _momentum(
                _edge,
                _minkowski(
                    4, _source_index(_index_edge, _shift)
                ),
            ),
        )
        converted = converted.replace(
            _momentum(_edge, _sink_index(_index_edge, _shift)),
            _momentum(
                _edge,
                _minkowski(4, _sink_index(_index_edge, _shift)),
            ),
        )

        contracted = simplify_metrics(converted.expand())
        contracted = simplify_gamma(contracted)
        contracted = simplify_color(contracted)
        return to_dots(simplify_metrics(contracted.expand()))

    def project_gluon_self_energy_scalar(
        expression: Expression,
    ) -> Expression:
        """Apply the normalized transverse and color-singlet projector."""
        _mu = _sink_index(0, 1)
        _nu = _sink_index(1, 1)
        _color_a = _color_index(0, 1)
        _color_b = _color_index(1, 1)
        _external_momentum = _momentum(0, _minkowski(4))
        _momentum_squared = _dot(
            _external_momentum,
            _external_momentum,
        )
        _color_average = _metric(
            _adjoint(8, _color_a),
            _adjoint(8, _color_b),
        ) / 8
        _transverse_average = (
            _metric(_minkowski(4, _mu), _minkowski(4, _nu))
            - _momentum(0, _minkowski(4, _mu))
            * _momentum(0, _minkowski(4, _nu))
            / _momentum_squared
        ) / 3

        projected = simplify_metrics(
            (expression * _color_average * _transverse_average).expand()
        )
        projected = simplify_color(projected)
        return to_dots(simplify_metrics(projected.expand()))

    return contract_qcd_numerator, project_gluon_self_energy_scalar


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project the contracted tensor to a diagnostic scalar

    FeynKit keeps the model's UFO tensor heads and gives every propagator
    endpoint a stable `SourceIndex` or `SinkIndex`. Before contraction, the
    symbolic adapter above assigns their physical spenso representations:
    four-dimensional Minkowski and bispinor slots, plus SU(3) adjoint and
    fundamental slots. The color labels at both ends of an internal edge are
    identified because the propagator's color delta is implicit. The adapter
    also resolves
    \(\not{p}=\gamma^\rho p_\rho\) with one private Lorentz index per
    propagator.

    The actual algebra is then native idenso: metrics sew the propagator
    endpoints, `simplify_gamma` closes the bottom-quark Dirac trace, and
    `simplify_color` contracts the two structure constants or generators.
    This first stage produces the tensor
    \(N^{ab}_{\mu\nu}\) with only the two external-gluon index pairs free.

    For a scalar diagnostic at generic off-shell \(p^2\ne0\), the next stage
    applies the normalized 4D transverse, SU(3) color-singlet projector

    \[
    \mathcal P^{ab}_{\mu\nu}
      = \frac{\delta^{ab}}{8}\,
        \frac{1}{3}\left(g_{\mu\nu}
          - \frac{p_\mu p_\nu}{p^2}\right),
      \qquad p=\operatorname{Momentum}(0).
    \]

    Its normalization is
    \(1/[(d-1)(N_c^2-1)]=1/(3\cdot8)=1/24\). Native spenso metrics
    identify and contract the external Lorentz and adjoint slots; idenso then
    simplifies them and rewrites every momentum contraction as a symmetric
    `spenso::dot`. The displayed result is therefore a scalar with no
    `SinkIndex` or `ColorIndex` labels.

    This is a projection of the selected **numerator**, not the unprojected
    self-energy tensor or a claim that each loop class is separately
    transverse. Although the gluon field is massless, the standard projector
    above treats its self-energy momentum as off-shell; it is singular at
    \(p^2=0\). Individual contributions need not be transverse before the
    appropriate gauge-sector sum and loop integration. Changing the graph
    selector reruns both contraction stages for that loop.
    """)
    return


@app.cell
def _(
    contract_qcd_numerator,
    mo,
    project_gluon_self_energy_scalar,
    selected_diagram,
    selected_label,
    table,
):
    contracted_numerator = contract_qcd_numerator(
        selected_diagram.numerator_expression()
    )
    scalar_projection = project_gluon_self_energy_scalar(
        contracted_numerator
    )

    table(
        [
            {
                "contribution": selected_label,
                "normalized projector": mo.md(
                    r"""\(\frac{\delta^{ab}}{8}\frac{1}{3}
                    (g_{\mu\nu}-p_\mu p_\nu/p^2)\)"""
                ),
                "projected numerator": mo.as_html(scalar_projection),
                "result": "scalar; no free Lorentz or color indices",
            }
        ],
        column_widths={
            "contribution": 180,
            "normalized projector": 350,
            "projected numerator": 760,
            "result": 270,
        },
    )
    return contracted_numerator, scalar_projection


if __name__ == "__main__":
    app.run()
