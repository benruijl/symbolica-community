"""Regression coverage for structured Spenso composition and display boundaries."""

import pytest
from symbolica import S
from symbolica.community.spenso import (
    _,
    BroadcastFunction,
    Representation,
    Tensor,
    TensorFunctionLibrary,
    TensorLibrary,
    TensorName,
    TensorNetwork,
    as_tensor,
    chain,
    trace,
)


def execute(expression, library, functions=None):
    assert expression.name is not None
    network = TensorNetwork(expression, library=library)
    network.execute(library=library, function_library=functions)
    return network.result_tensor(library=library)


def test_scalar_metadata_is_opaque_to_indexing_and_materialization():
    representation = Representation.euc(2)
    index = representation("hardening_metadata_mu")
    metadata = S("hardening_metadata_value")(representation.to_expression())
    tensor = TensorName("hardening_metadata_tensor")

    indexed = tensor(metadata, representation).index(index)
    network = TensorNetwork(indexed)

    assert indexed.rank == 1
    assert indexed.interface == (index,)
    assert "hardening_metadata_value" in indexed.to_expression().to_canonical_string()
    assert "hardening_metadata_value" in str(network)


def test_structured_zero_operations_preserve_interface_metadata_and_execute():
    euc = Representation.euc(2)
    mink = Representation.mink(2)
    bis = Representation.bis(2)
    mu = euc("hardening_zero_mu")
    p = TensorName.vector("hardening_zero_p")
    q = TensorName.vector("hardening_zero_q")
    matrix = TensorName("hardening_zero_matrix")
    zero = (p(euc) * 0).with_name(TensorName("hardening_zero_result"))

    assert zero.index(mu).interface == (mu,)
    assert zero.contract(q(euc), left=0, right=0).is_scalar
    assert (zero * q(euc)).is_scalar
    assert zero.outer(q(mink)).rank == 2
    assert (matrix(bis, bis) * 0).trace().is_scalar

    network = TensorNetwork(zero)
    network.execute()
    assert network.result_tensor()[:] == pytest.approx([0.0, 0.0])
    scalar_zero = (p(euc) * p(euc)).with_name(TensorName("hardening_scalar_zero"))
    assert Tensor.dense(scalar_zero, [0.0])[:] == pytest.approx([0.0])


def test_explicit_index_normalization_and_automatic_outer_fallback():
    representation = Representation.euc(2)
    mu = representation("hardening_explicit_mu")
    tensor = TensorName("hardening_explicit_tensor")
    vector = TensorName.vector("hardening_explicit_vector")

    assert as_tensor(tensor(mu, mu).to_expression()).is_scalar
    triple = (
        vector(mu).to_expression()
        * vector(mu).to_expression()
        * vector(mu).to_expression()
    )
    with pytest.raises(ValueError, match="more than two"):
        as_tensor(triple)
    with pytest.raises(ValueError, match="outer product cannot preserve"):
        (vector(mu) * 1).outer(vector(mu))

    left = tensor(representation("hardening_i"), representation("hardening_j")) * 1
    right = tensor(representation("hardening_k"), representation("hardening_l")) * 1
    assert (left * right).rank == 4
    with pytest.raises(ValueError, match="same abstract index"):
        left.compose(right, left=(0, 1), right=(0, 1))


def test_powers_and_tensor_scalar_wrappers_execute_through_dot_and_chain():
    euc = Representation.euc(2)
    mink = Representation.mink(2)
    bis = Representation.bis(2)
    p = TensorName.vector("hardening_power_p")
    q = TensorName.vector("hardening_power_q")
    library = TensorLibrary()
    library.register(Tensor.dense(p(euc), [3.0, 4.0]))
    library.register(Tensor.dense(q(euc), [1.0, 2.0]))

    power = as_tensor(as_tensor(p(euc)).to_expression() ** 2).with_name(
        TensorName("hardening_power_result")
    )
    assert "dot" in power.to_expression().to_canonical_string()
    assert str(execute(power, library)[0]).startswith("25")

    scalar = p(euc) * p(euc)
    nested_dot = ((scalar * q(euc)) * q(euc)).with_name(
        TensorName("hardening_nested_dot_result")
    )
    assert str(execute(nested_dot, library)[0]).startswith("125")

    gamma = TensorName.gamma()(mink, bis, bis)
    factors = [
        gamma.index(mink(f"hardening_chain_{name}"), _, _)
        for name in ("mu", "nu", "rho")
    ]
    extended = (scalar * (factors[0] * factors[1])) * factors[2]
    assert extended.rank == 5
    TensorNetwork(extended)


def test_broadcast_callbacks_cover_vectors_and_tensor_derived_scalars():
    representation = Representation.euc(2)
    p = TensorName.vector("hardening_callback_p")
    q = TensorName.vector("hardening_callback_q")
    identity = BroadcastFunction("hardening_callback_identity")
    sqrt = BroadcastFunction("hardening_callback_sqrt")
    library = TensorLibrary()
    p_expression = as_tensor(p(representation))
    q_expression = as_tensor(q(representation))
    library.register(Tensor.dense(p_expression, [3.0, 4.0]))
    library.register(Tensor.dense(q_expression, [1.0, 2.0]))
    functions = TensorFunctionLibrary()
    functions.register(identity, lambda value: value)
    functions.register(sqrt, lambda value: value**0.5)

    wrapped_dot = (identity(p_expression) * q_expression).with_name(
        TensorName("hardening_wrapped_dot_result")
    )
    assert str(execute(wrapped_dot, library, functions)[0]).startswith("11")

    scalar = sqrt(p_expression * p_expression).with_name(
        TensorName("hardening_sqrt_result")
    )
    with pytest.raises(RuntimeError, match="no concrete callback registered"):
        execute(scalar, library, TensorFunctionLibrary())
    assert str(execute(scalar, library, functions)[0]).startswith("5")


def test_identity_helpers_and_concrete_display_use_semantic_ordering():
    euc = Representation.euc(2)
    identity = chain(euc("hardening_a"), euc("hardening_b"))
    dimension = trace(euc)

    assert identity.rank == 2
    TensorNetwork(identity)
    assert dimension.is_scalar
    assert "Tr" in dimension.format_tensor()
    TensorNetwork(dimension)

    color = Representation.cof(2)
    with pytest.raises(ValueError, match="input-to-output"):
        chain(color.dual()("hardening_ci"), color("hardening_cj"))

    ordered = Tensor.dense(
        TensorName("hardening_ordered")(euc(2), euc(1)),
        [0.0, 1.0, 2.0, 3.0],
    )
    assert ordered[:] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert ordered.format_tensor().endswith("[0, 1]\n[2, 3]")

    mixed_descriptor = TensorName("hardening_mixed")(
        Representation.mink(2)("hardening_m"),
        Representation.euc(3)("hardening_e"),
    )
    assert mixed_descriptor[4] == [1, 1]
    assert mixed_descriptor[[1, 2]] == 5
    mixed = Tensor.dense(mixed_descriptor, list(range(6)))
    assert mixed[:] == pytest.approx(list(range(6)))
    assert mixed[[1, 2]] == pytest.approx(5.0)
    assert mixed.format_tensor().endswith("[0, 1, 2]\n[3, 4, 5]")

    sparse = Tensor.sparse(
        TensorName.vector("hardening_sparse")(Representation.euc(4)("hardening_s")),
        float,
    )
    sparse[2] = 7.0
    assert sparse.format_tensor().endswith("[0, 0, 7, 0]")


def test_latex_scripts_use_latex_grouping_without_affecting_typst():
    mink = Representation.mink(2)
    bis = Representation.bis(2)
    gamma = TensorName.gamma()(mink, bis, bis).index(mink("hardening_latex_mu"), _, _)
    latex = gamma._repr_latex_()

    assert r"\gamma^{" in latex
    assert r"\gamma^(" not in latex
    assert "^(" in gamma.to_typst()

    metric_latex = as_tensor(
        mink.g("hardening_latex_i", "hardening_latex_j")
    )._repr_latex_()
    assert "_(" not in metric_latex
    assert "^(" not in metric_latex
