import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from functools import partial
    import math
    from pathlib import Path

    import marimo as mo
    import symbolica.community.feynkit as fk
    from symbolica import S

    table = partial(mo.ui.table, selection=None)
    data_file = Path(__file__).resolve().parent / "data" / "scalars_2p_3p.json"
    return S, data_file, fk, math, mo, table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A one-loop diagram at local loop-momentum points

    This tutorial follows one scalar one-loop graph from generation through
    its native Feynman-rule expressions and Cross-Free Family (CFF)
    denominators. We then route two concrete loop momenta through the native
    momentum basis and evaluate the resulting CFF surfaces. The final numbers
    probe the local denominator structure; they are not cross sections or
    loop-integrated amplitudes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate exactly one loop

    The normalized teaching model contains three neutral scalars. We retain
    only the massless \(\phi_0^3\) interaction and request loop order one.
    Self-loop topologies are generated, but a diagram without a self-edge is
    selected so its momentum routing is especially easy to inspect.
    """)
    return


@app.cell
def _(data_file, fk, mo, table):
    model = fk.Model.from_path(data_file)
    _options = fk.GenerationOptions(max_vertices=3, allow_self_loops=True)
    _options.add_vertex_allow(["V_3_SCALAR_000"])
    _generated = model.generate_diagrams(
        incoming=["scalar_0"],
        outgoing=["scalar_0", "scalar_0"],
        loops=1,
        options=_options,
    )

    diagram = next(
        item
        for item in _generated.diagrams
        if all(edge.source != edge.target for edge in item.edges)
    )
    diagram.validate(model)
    basis = diagram.loop_momentum_bases(limit=1)[0]

    table(
        [
            {
                "diagram": mo.as_html(diagram),
                "name": diagram.name,
                "loops": diagram.loop_count,
                "symmetry factor": diagram.symmetry_factor,
            }
        ],
        column_widths={"diagram": 460},
    )
    return basis, diagram, model


@app.cell
def _(diagram):
    diagram
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the native momentum routing

    A `LoopMomentumBasis` expresses every edge momentum in terms of the
    independent loop momentum \(k_0\) and external momenta \(p_i\). The
    coefficients are integers fixed by graph momentum conservation.
    """)
    return


@app.cell
def _(basis, diagram, mo, table):
    _edges = {edge.id: edge for edge in diagram.edges}
    _routing_rows = [
        {
            "edge": edge_id,
            "particle": _edges[edge_id].particle_name,
            "role": (
                "loop"
                if edge_id in basis.loop_edges
                else "external"
                if edge_id in basis.external_edges
                else "tree"
            ),
            "momentum": signature.format_momentum(),
        }
        for edge_id, signature in basis.edge_signatures.items()
    ]

    mo.vstack(
        [
            mo.as_html(basis),
            table(_routing_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect native Feynman-rule expressions

    Internal vertices and propagator edges expose their numerator annotations
    as native Symbolica `Expression` objects. For this scalar model, each
    vertex contributes \(i\lambda\), every propagator numerator is one, and
    the diagram numerator is their product, \(-i\lambda^3\). The graph-wide
    factor deliberately keeps `AutG` and the external-fermion ordering sign as
    symbolic atoms; both evaluate to one for this scalar graph, and the code
    supplies those values explicitly.
    """)
    return


@app.cell
def _(S, diagram, mo, model, table):
    _lam_value = model.parameter("lam").value
    _rule_point = {S("UFO::lam"): _lam_value}
    _rule_rows = []
    for _vertex in (item for item in diagram.vertices if not item.is_external):
        _rule_expression = _vertex.numerator_expression()
        _model_rule = model.vertex_rule(_vertex.interaction)
        _rule_rows.append(
            {
                "object": f"vertex {_vertex.id}",
                "model rule": _vertex.interaction,
                "couplings": _model_rule.couplings,
                "numerator": mo.as_html(_rule_expression.formatted()),
                "value at lambda=1": _rule_expression.evaluate(_rule_point),
            }
        )
    for _edge in diagram.edges:
        _rule_expression = _edge.numerator_expression()
        _rule_rows.append(
            {
                "object": f"edge {_edge.id}",
                "model rule": f"{_edge.particle_name} propagator",
                "couplings": None,
                "numerator": mo.as_html(_rule_expression.formatted()),
                "value at lambda=1": _rule_expression.evaluate(_rule_point),
            }
        )

    diagram_numerator = diagram.numerator_expression()
    _overall_factor = diagram.overall_factor_expression()
    numerator_value = diagram_numerator.evaluate(_rule_point)
    overall_value = _overall_factor.evaluate(
        {
            S("feynkit_py::AutG")(diagram.symmetry_factor): float(
                diagram.symmetry_factor
            ),
            S("feynkit_py::ExternalFermionOrderingSign")(1): 1.0,
        }
    )
    mo.vstack(
        [
            table(
                _rule_rows,
                column_widths={"numerator": 190},
            ),
            table(
                [
                    {
                        "diagram numerator": mo.as_html(
                            diagram_numerator.formatted()
                        ),
                        "numerator value": numerator_value,
                        "overall factor": mo.as_html(_overall_factor.formatted()),
                        "factor value": overall_value,
                    }
                ]
            ),
        ]
    )
    return diagram_numerator, numerator_value, overall_value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert the graph to a Cross-Free Family

    CFF rewrites the loop-energy structure as sums of products of causal
    surfaces. The typed surfaces retain the internal edge energies, signed
    external-energy shift, and enclosed vertices used to define each formal
    symbol \(E_i\).
    """)
    return


@app.cell
def _(diagram, mo, table):
    cff = diagram.build_cff(max_orientations=10_000)
    cff_expression = cff.to_expression()
    _report = cff.report

    table(
        [
            {
                "CFF expression": mo.as_html(cff_expression.formatted()),
                "acyclic orientations": _report.acyclic_orientations,
                "terms": _report.unfolded_terms,
                "surfaces": _report.interned_surfaces,
            }
        ],
        column_widths={"CFF expression": 430},
    )
    return cff, cff_expression


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose two nonsingular loop-momentum points

    Use arbitrary but consistent energy units. The external massless momenta
    obey \(p_0=p_1+p_2\). We inspect the loop three-vectors
    \(\mathbf{k}_A=(1,2,3)\) and \(\mathbf{k}_B=(-2,1,2)\). These are local
    denominator probes, not a loop integration or phase-space sampling.
    """)
    return


@app.cell
def _(fk, mo, table):
    _incoming = fk.FourMomentum(10.0, 0.0, 0.0, 10.0)
    _outgoing_1 = fk.FourMomentum(4.0, 0.0, 0.0, 4.0)
    _outgoing_2 = fk.FourMomentum(6.0, 0.0, 0.0, 6.0)
    external_momenta = [_incoming, _outgoing_1, _outgoing_2]
    loop_points = {
        "A": fk.ThreeMomentum(1.0, 2.0, 3.0),
        "B": fk.ThreeMomentum(-2.0, 1.0, 2.0),
    }

    table(
        [
            {
                "momentum": label,
                "components": mo.as_html(momentum),
            }
            for label, momentum in (
                ("incoming p0", _incoming),
                ("outgoing p1", _outgoing_1),
                ("outgoing p2", _outgoing_2),
                ("loop point A", loop_points["A"]),
                ("loop point B", loop_points["B"]),
            )
        ]
    )
    return external_momenta, loop_points


@app.cell
def _(basis, external_momenta, loop_points, math, mo, table):
    _external_spatial = [
        (momentum.px, momentum.py, momentum.pz)
        for momentum in external_momenta
    ]
    _loop_spatial_by_point = {
        point: [(momentum.px, momentum.py, momentum.pz)]
        for point, momentum in loop_points.items()
    }
    _routed_spatial_by_point = {}
    on_shell_energy_by_point = {}
    for _point, _loop_spatial in _loop_spatial_by_point.items():
        _routed_spatial = {}
        for _edge_id, _signature in basis.edge_signatures.items():
            _routed_spatial[_edge_id] = tuple(
                sum(
                    coefficient * vector[axis]
                    for coefficient, vector in zip(
                        _signature.loops, _loop_spatial
                    )
                )
                + sum(
                    coefficient * vector[axis]
                    for coefficient, vector in zip(
                        _signature.external, _external_spatial
                    )
                )
                for axis in range(3)
            )
        _routed_spatial_by_point[_point] = _routed_spatial
        on_shell_energy_by_point[_point] = {
            edge_id: math.sqrt(
                sum(component * component for component in _routed_spatial[edge_id])
            )
            for edge_id in basis.tree_edges + basis.loop_edges
        }

    external_energy_by_edge = {
        edge_id: external_momenta[index].energy
        for index, edge_id in enumerate(basis.external_edges)
    }
    table(
        [
            {
                "point": point,
                "internal edge": edge_id,
                "routed spatial momentum": tuple(
                    round(component, 6)
                    for component in _routed_spatial_by_point[point][edge_id]
                ),
                "on-shell energy": round(energy, 12),
            }
            for point, energies in on_shell_energy_by_point.items()
            for edge_id, energy in sorted(energies.items())
        ]
    )
    return external_energy_by_edge, on_shell_energy_by_point


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate the formal CFF expression

    FeynKit currently exposes the exact surface definitions but no native
    momentum-to-surface evaluator or `CffResult.evaluate(...)` method. We
    therefore use each native `MomentumSignature` above to route the points,
    then inspect every surface explicitly:

    \[
      E_i=\sum_{e\in +}\omega_e-\sum_{e\in -}\omega_e
          +\sum_{a}c_a p_a^0,
      \qquad \omega_e=\sqrt{\mathbf q_e^2+m_e^2}.
    \]

    This teaching model has massless internal `scalar_0` lines, so the routed
    on-shell energies are \(\omega_e=|\mathbf q_e|\). Once the six surface
    values are assembled at each point, Symbolica evaluates the native CFF
    expression directly.
    """)
    return


@app.cell
def _(
    S,
    cff,
    cff_expression,
    diagram_numerator,
    external_energy_by_edge,
    mo,
    numerator_value,
    on_shell_energy_by_point,
    overall_value,
    table,
):
    _surface_symbol_values = {}
    _surface_rows = []
    for _surface in cff.surfaces:
        _value = sum(
            on_shell_energy_by_point["A"][edge]
            for edge in _surface.positive_energies
        ) - sum(
            on_shell_energy_by_point["A"][edge]
            for edge in _surface.negative_energies
        ) + sum(
            coefficient * external_energy_by_edge[edge]
            for edge, coefficient in _surface.external_shift
        )
        _surface_symbol_values[S(_surface.symbol_name)] = _value
        _surface_rows.append(
            {
                "surface": _surface.symbol_name,
                "positive energies": _surface.positive_energies,
                "negative energies": _surface.negative_energies,
                "external shift": _surface.external_shift,
                "value": round(_value, 9),
            }
        )

    cff_value = cff_expression.evaluate(_surface_symbol_values)
    _point_rows = []
    _cff_values = {}
    for _point, _on_shell_energies in on_shell_energy_by_point.items():
        _symbol_values = {}
        for _surface in cff.surfaces:
            _symbol_values[S(_surface.symbol_name)] = sum(
                _on_shell_energies[edge] for edge in _surface.positive_energies
            ) - sum(
                _on_shell_energies[edge] for edge in _surface.negative_energies
            ) + sum(
                coefficient * external_energy_by_edge[edge]
                for edge, coefficient in _surface.external_shift
            )
        _cff_values[_point] = cff_expression.evaluate(_symbol_values)
        _point_rows.append(
            {
                "point": _point,
                "loop momentum": tuple(
                    round(component, 6)
                    for component in (
                        (1.0, 2.0, 3.0)
                        if _point == "A"
                        else (-2.0, 1.0, 2.0)
                    )
                ),
                "CFF value": _cff_values[_point],
            }
        )

    mo.vstack(
        [
            table(_surface_rows),
            table(_point_rows),
            table(
                [
                    {
                        "diagram numerator": mo.as_html(
                            diagram_numerator.formatted()
                        ),
                        "numerator value": numerator_value,
                        "overall factor value": overall_value,
                        "CFF expression": mo.as_html(cff_expression.formatted()),
                        "CFF value": cff_value,
                    }
                ],
                column_widths={
                    "diagram numerator": 180,
                    "CFF expression": 390,
                },
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation and next steps

    The table keeps the graph bookkeeping factor, native Feynman-rule
    numerator, and local CFF denominator probes separate. Multiplying those
    entries would still not create a full integrand: the loop measure, contour
    or \(i0\) prescription, integration, renormalization, and observable
    normalization are deliberately absent.

    For production work, a dedicated evaluator should own the momentum map,
    particle masses, prescriptions, and numerical stability checks. Until
    FeynKit exposes that layer natively, the explicit surface table above is
    the closest auditable local-point inspection supported by the public API.
    """)
    return


if __name__ == "__main__":
    app.run()
