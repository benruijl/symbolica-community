"""Focused integration checks for the structured Spenso expression API."""

import re

import pytest
from symbolica import Expression, S
from symbolica.community.spenso import (
    _,
    AUTO,
    BroadcastFunction,
    Representation,
    Slot,
    TensorExpression,
    TensorName,
    as_tensor,
)


def test_tensor_expression_lifecycle_and_local_auto_ports():
    mink = Representation.mink(4)
    bis = Representation.bis(4)
    mu = mink("mu")
    gamma = TensorName.gamma()(mink, bis, bis)

    open_gamma = gamma(mu, _, AUTO)
    assert isinstance(open_gamma, TensorExpression)
    assert isinstance(open_gamma, Expression)
    assert open_gamma.rank == 3
    assert not open_gamma.is_scalar
    assert len(open_gamma.interface) == 3

    indexed_gamma = open_gamma("i", "j")
    assert isinstance(indexed_gamma, TensorExpression)
    assert all(isinstance(port, Slot) for port in indexed_gamma.interface)

    restored = as_tensor(open_gamma.to_expression())
    assert isinstance(open_gamma.to_expression(), Expression)
    assert not isinstance(open_gamma.to_expression(), TensorExpression)
    assert isinstance(restored, TensorExpression)
    assert restored.rank == open_gamma.rank


def test_vector_dot_chain_trace_and_outer_return_structured_expressions():
    mink = Representation.mink(4)
    bis = Representation.bis(4)
    mu = mink("mu")
    nu = mink("nu")
    p = TensorName.vector("structured_api_p", is_linear=True)
    gamma = TensorName.gamma()(mink, bis, bis)

    square = p(1, mink) * p(1, mink)
    assert isinstance(square, TensorExpression)
    assert square.is_scalar
    assert square.rank == 0
    assert "dot" in square.to_expression().to_canonical_string()

    line = gamma.index(mu, _, _) * gamma.index(nu, _, _)
    assert isinstance(line, TensorExpression)
    assert line.rank == 4
    assert "chain" in line.to_expression().to_canonical_string()
    assert [str(port) for port in line.interface if isinstance(port, Slot)] == [
        str(mu),
        str(nu),
    ]

    traced = line.trace()
    assert isinstance(traced, TensorExpression)
    assert traced.rank == 2
    assert "trace" in traced.to_expression().to_canonical_string()
    assert [str(port) for port in traced.interface] == [str(mu), str(nu)]

    placeholder = re.compile(r"(?<![A-Za-z0-9_])(in|out)(?![A-Za-z0-9_])")
    for expression in (line, traced):
        for rendered in (
            repr(expression),
            str(expression),
            expression.format_tensor(),
            expression.to_typst(),
            str(expression.formatted()),
        ):
            assert placeholder.search(rendered) is None
            assert rendered.find("mu") < rendered.find("nu")

    outer = p(1, mink).outer(p(2, mink))
    assert isinstance(outer, TensorExpression)
    assert outer.rank == 2


def test_addition_requires_the_same_interface():
    mink = Representation.mink(4)
    p = TensorName.vector("structured_api_add_p")
    q = TensorName.vector("structured_api_add_q")

    assert isinstance(p(1, mink) + q(1, mink), TensorExpression)
    with pytest.raises(ValueError, match="interface"):
        _ = p(1, mink("mu")) + q(1, mink("nu"))


def test_tags_broadcast_and_semantic_formatting_preserve_the_interface():
    mink = Representation.mink(4)
    p = TensorName.vector(
        "structured_api_broadcast_p", is_linear=True, tags=["kinematics"]
    )
    sqrt = BroadcastFunction("structured_api_sqrt", is_real=True, tags=["kinematics"])
    vector = p(S("x"), mink)
    result = sqrt(vector)

    assert p.has_tag("spenso::tensor")
    assert p.has_tag("spenso::rank1")
    assert p.has_tag("kinematics")
    assert sqrt.has_tag("spenso::broadcast")
    assert not sqrt.has_tag("spenso::tensor")
    assert set(p.get_tags()) >= {
        "python::kinematics",
        "spenso::tensor",
        "spenso::rank1",
    }
    assert set(sqrt.get_tags()) >= {"python::kinematics", "spenso::broadcast"}
    assert isinstance(result, TensorExpression)
    assert result.interface == vector.interface

    canonical = result.to_expression().to_canonical_string()
    assert isinstance(result.format_tensor(), str)
    assert isinstance(result.to_typst(), str)
    assert str(result.formatted()) == result.format_tensor()
    assert result.to_expression().to_canonical_string() == canonical


def test_rank_one_invariant_and_rank_validation():
    mink = Representation.mink(4)
    vector = TensorName.vector("structured_api_rank_p")

    with pytest.raises(ValueError, match="exactly one"):
        vector(1)
    with pytest.raises(ValueError, match="exactly one"):
        vector(mink, mink)
    with pytest.raises(ValueError, match="precede"):
        vector(mink, 1)
    with pytest.raises(ValueError, match="rank"):
        TensorName("structured_api_rank_bad", rank=2)
