"""End-to-end checks for the installed native FeynKit community module."""

import inspect
from pathlib import Path

import pytest

MODEL_PATH = (
    Path(__file__).parents[1] / "examples" / "feynkit" / "data" / "scalars_2p_3p.json"
)
SM_MODEL_PATH = MODEL_PATH.with_name("sm.json")


def test_feynkit_and_spenso_share_symbolica():
    """Both modules exchange expressions through the installed core module."""

    import sys

    from symbolica import Expression
    from symbolica.community import feynkit, spenso

    assert "symbolica.community.feynkit_native" in sys.modules
    assert "_gammaloop" not in sys.modules
    for exported in (
        feynkit.Model,
        feynkit.Generator,
        feynkit.FeynmanDiagram,
        feynkit.CffGenerator,
        feynkit.FourMomentum,
    ):
        assert exported.__module__ == "symbolica.community.feynkit"

    model = feynkit.Model(MODEL_PATH)
    scalar = model.particle("scalar_0")
    options = feynkit.GenerationOptions(max_vertices=3)
    options.add_vertex_allow(["V_3_SCALAR_000"])
    options.add_particle_veto([model.particle("scalar_1"), 1002])
    generated = model.generate_diagrams(
        [scalar],
        [scalar, scalar.antiparticle],
        options=options,
    )
    factor = generated.diagrams[0].overall_factor_expression()
    tensor = spenso.TensorName("T").to_expression()
    assert type(factor) is Expression
    assert type(factor + tensor) is Expression


def test_particle_antiparticle_and_native_particle_inputs():
    """Particles retain their model relation and normalize in process inputs."""

    import symbolica.community.feynkit as fk

    assert not hasattr(fk.Model, "from_path")

    model = fk.Model(SM_MODEL_PATH)
    bottom = model.particle("b")
    assert isinstance(bottom.antiparticle, fk.Particle)
    assert (bottom.antiparticle.name, bottom.antiparticle.pdg_code) == ("b~", -5)
    assert bottom.antiparticle.antiparticle.name == bottom.name

    gluon = model.particle("g")
    assert gluon.antiparticle.name == gluon.name
    process = fk.Process.amplitude([gluon], [gluon.antiparticle])
    assert process.incoming[0].pdg == 21
    assert process.outgoing_alternatives[0][0].pdg == 21


def test_generated_ufo_tensors_are_native_spenso_expressions():
    """Generated SM numerators can enter Idenso without a Python adapter."""

    import symbolica.community.feynkit as fk
    from symbolica.community.idenso import (
        simplify_color,
        simplify_gamma,
        simplify_metrics,
        to_dots,
    )

    model = fk.Model(SM_MODEL_PATH)
    gluon = model.particle("g")
    options = fk.GenerationOptions(max_vertices=2)
    options.set_coupling_orders({"QCD": (2, 2), "QED": (0, 0)})
    options.add_particle_veto(
        [
            particle
            for name in ("d", "u", "s", "c", "t")
            for quark in (model.particle(name),)
            for particle in (quark, quark.antiparticle)
        ]
    )
    generated = model.generate_diagrams(
        incoming=[gluon],
        outgoing=[gluon.antiparticle],
        loops=1,
        options=options,
    )

    assert len(generated.diagrams) == 3
    serialized = "\n".join(
        repr(diagram.numerator_expression()) for diagram in generated.diagrams
    )
    assert "UFO::GC_" in serialized
    for head in ("Gamma", "Metric", "PSlash", "Identity", "T", "f"):
        assert f"UFO::{head}(" not in serialized
    for head in ("gamma", "t", "f", "g"):
        assert f"spenso::{head}(" in serialized

    for diagram in generated.diagrams:
        expression = model.expand_couplings(diagram.numerator_expression())
        assert "UFO::GC_" not in repr(expression)
        assert "UFO::G" in repr(expression)
        expression = simplify_metrics(expression.expand())
        expression = simplify_gamma(expression)
        expression = simplify_color(expression)
        to_dots(simplify_metrics(expression.expand()))


def test_feynkit_owner_api_covers_cff_and_jets():
    """Physics operations live on the model, diagram, and jet definition."""

    import symbolica.community.feynkit as fk
    from symbolica import Expression

    model = fk.Model(MODEL_PATH)
    options = fk.GenerationOptions(max_vertices=3, allow_self_loops=True)
    options.add_vertex_allow(["V_3_SCALAR_000"])
    generated = model.generate_diagrams(
        ["scalar_0"],
        [1000, "scalar_0"],
        loops=(0, 1),
        options=options,
    )
    loop_diagram = next(
        diagram
        for diagram in generated.diagrams
        if diagram.loop_count == 1
        and all(edge.source != edge.target for edge in diagram.edges)
    )
    assert isinstance(loop_diagram.build_cff().to_expression(), Expression)

    clustered = fk.JetDefinition.anti_kt(radius=0.4).cluster(
        [
            fk.FourMomentum(10.0, 10.0, 0.0, 0.0),
            fk.FourMomentum(5.0, 5.0, 0.0, 0.0),
        ]
    )
    assert len(clustered) == 1
    assert clustered[0].constituent_indices == [0, 1]

    assert model.parameter("lam").value == complex(1.0, 0.0)
    assert model.coupling("SCALAR_COUPLING").value == complex(0.0, 1.0)


def test_owner_workflows_are_native_extension_methods():
    """The owner-oriented workflows are PyO3 descriptors, not Python wrappers."""

    import symbolica.community.feynkit as fk

    for removed in (
        "ParticleLike",
        "LoopOrder",
        "load_ufo_model",
        "generate_diagrams",
        "build_cff",
        "cluster_jets",
    ):
        assert not hasattr(fk, removed)

    for owner, method_name in (
        (fk.Model, "generate_diagrams"),
        (fk.Model, "expand_couplings"),
        (fk.FeynmanDiagram, "build_cff"),
        (fk.JetDefinition, "cluster"),
        (fk.UfoLoader, "load"),
    ):
        method = getattr(owner, method_name)
        assert inspect.ismethoddescriptor(method)
        assert method.__objclass__ is owner

    assert inspect.isbuiltin(fk.JetDefinition.anti_kt)


def test_model_generation_rejects_ambiguous_configuration():
    """The model method rejects ambiguous process configuration early."""

    import symbolica.community.feynkit as fk

    model = fk.Model(MODEL_PATH)
    with pytest.raises(ValueError, match="kind must be"):
        model.generate_diagrams(["scalar_0"], ["scalar_0"], kind="rate")
    with pytest.raises(ValueError, match="loop bounds"):
        model.generate_diagrams(["scalar_0"], ["scalar_0"], loops=(2, 1))
    with pytest.raises(TypeError, match="loops must be"):
        model.generate_diagrams(["scalar_0"], ["scalar_0"], loops=True)
    with pytest.raises(ValueError, match="final_state_alternatives"):
        model.generate_diagrams(
            ["scalar_0"],
            ["scalar_0"],
            final_state_alternatives=[["scalar_0"]],
        )
