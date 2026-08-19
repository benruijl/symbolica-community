"""Runtime, stub, and structured-dispatch parity for Idenso."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from symbolica import Expression, S
from symbolica.community import idenso, idenso_native
from symbolica.community.spenso import (
    Representation,
    TensorExpression,
    TensorName,
    as_tensor,
)


SETTINGS_AND_ERRORS = {
    "CanonicalizationError",
    "ColorCasimirSettings",
    "ColorSimplifySettings",
    "CookMode",
    "CookSettings",
    "CookSourceFilter",
    "CookTagFilter",
    "CookingError",
    "DiracAdjointError",
    "DotExpansionError",
    "GammaChainOrdering",
    "GammaConjugationError",
    "GammaSimplifySettings",
    "SchoonschipContractionOrder",
    "SchoonschipMode",
    "SchoonschipSettings",
    "SchoonschipTraversal",
}


def _stub_tree() -> ast.Module:
    return ast.parse(Path(idenso.__file__).with_name("__init__.pyi").read_text())


def _stub_exports() -> set[str]:
    return {
        node.name
        for node in _stub_tree().body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def _stub_functions() -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in _stub_tree().body
        if isinstance(node, ast.FunctionDef)
    }


def _stub_tensor_methods() -> dict[str, ast.FunctionDef]:
    path = Path(idenso.__file__).parent.parent / "spenso" / "__init__.pyi"
    tree = ast.parse(path.read_text())
    tensor_expression = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TensorExpression"
    )
    return {
        node.name: node
        for node in tensor_expression.body
        if isinstance(node, ast.FunctionDef)
    }


def _runtime_parameter_shape(value, *, drop_first: bool = False):
    parameters = list(inspect.signature(value).parameters.values())
    if drop_first:
        parameters = parameters[1:]
    return [
        (
            parameter.name,
            parameter.kind.name,
            parameter.default is not inspect.Parameter.empty,
        )
        for parameter in parameters
    ]


def _stub_parameter_shape(node: ast.FunctionDef, *, drop_first: bool = False):
    positional = [*node.args.posonlyargs, *node.args.args]
    required = len(positional) - len(node.args.defaults)
    parameters = [
        (
            argument.arg,
            (
                "POSITIONAL_ONLY"
                if index < len(node.args.posonlyargs)
                else "POSITIONAL_OR_KEYWORD"
            ),
            index >= required,
        )
        for index, argument in enumerate(positional)
    ]
    if node.args.vararg is not None:
        parameters.append((node.args.vararg.arg, "VAR_POSITIONAL", False))
    parameters.extend(
        (argument.arg, "KEYWORD_ONLY", default is not None)
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
    )
    if node.args.kwarg is not None:
        parameters.append((node.args.kwarg.arg, "VAR_KEYWORD", False))
    return parameters[1:] if drop_first else parameters


def test_runtime_stub_exports_and_method_inventory_match_exactly():
    runtime_exports = set(idenso_native.__all__)
    assert runtime_exports == _stub_exports()
    assert not {"initialize", "initialize_module", "NA", "rewrite_na"} & runtime_exports
    assert not hasattr(idenso, "initialize")
    assert not hasattr(idenso, "initialize_module")

    operations = runtime_exports - SETTINGS_AND_ERRORS
    assert operations
    assert all(hasattr(TensorExpression, operation) for operation in operations)
    assert all(not hasattr(Expression, operation) for operation in operations)


def test_every_runtime_and_stub_signature_and_return_type_match():
    operations = set(idenso_native.__all__) - SETTINGS_AND_ERRORS
    module_stubs = _stub_functions()
    method_stubs = _stub_tensor_methods()

    term_operations = {
        "expand_in_patterns",
        "expand_mink",
        "expand_bis",
        "expand_mink_bis",
        "expand_metrics",
        "expand_color",
    }
    term_return = "builtins.list[tuple[Expression, Expression]]"
    alias_return = "tuple[Expression, builtins.list[tuple[Expression, Expression]]]"

    for operation in operations:
        module = getattr(idenso, operation)
        method = getattr(TensorExpression, operation)
        module_stub = module_stubs[operation]
        method_stub = method_stubs[operation]

        assert _runtime_parameter_shape(module) == _stub_parameter_shape(module_stub)
        assert _runtime_parameter_shape(
            method, drop_first=True
        ) == _stub_parameter_shape(method_stub, drop_first=True)
        assert _runtime_parameter_shape(
            module, drop_first=True
        ) == _runtime_parameter_shape(method, drop_first=True)

        module_return = ast.unparse(module_stub.returns)
        method_return = ast.unparse(method_stub.returns)
        if operation in term_operations:
            assert module_return == method_return == term_return
        elif operation == "alias_subtensors":
            assert module_return == method_return == alias_return
        elif operation == "list_dangling":
            assert module_return == method_return == "builtins.list[Expression]"
        elif operation in {"cook", "cook_function"}:
            assert module_return == method_return == "Expression"
        else:
            assert module_return == "Expression"
            assert method_return == "TensorExpression"


def test_runtime_signatures_and_immutable_setting_defaults_match_rust():
    assert str(inspect.signature(idenso.simplify_gamma)) == "(self_, settings=None)"
    assert str(inspect.signature(idenso.to_color_casimir)) == (
        "(expression, *, fundamental, adjoint, settings=None)"
    )
    assert str(inspect.signature(TensorExpression.to_color_casimir)) == (
        "(self, /, *, fundamental, adjoint, settings=None)"
    )

    gamma = idenso.GammaSimplifySettings()
    assert gamma.chain_ordering == idenso.GammaChainOrdering.RepeatedPairs
    assert gamma.evaluate_traces is True
    assert gamma.expand_three_gamma_epsilon is False

    color = idenso.ColorSimplifySettings()
    assert color.evaluate_traces is True
    assert color.expand_cross_chain_fierz is True
    assert color.substitute_cof_dimension_invariants is False

    casimir = idenso.ColorCasimirSettings()
    assert casimir.rewrite_fundamental_dimension is True
    assert casimir.substitute_fundamental_index is False

    cook = idenso.CookSettings()
    assert cook.mode == idenso.CookMode.FlattenedSymbol
    assert cook.source_filter is not None
    assert cook.output_tags == []
    assert cook.preserve_tags is False
    assert idenso.CookSettings.indices().preserve_tags is True

    schoonschip = idenso.SchoonschipSettings()
    assert schoonschip.depth_limit == 1
    assert schoonschip.mode == idenso.SchoonschipMode.Recursive
    assert schoonschip.traversal == idenso.SchoonschipTraversal.BreadthFirst
    assert schoonschip.expand_contracted_sums is False
    assert schoonschip.simplify_chain_like_functions is False
    assert schoonschip.schoonschip_rank1_tensors is True
    assert (
        schoonschip.contraction_order
        == idenso.SchoonschipContractionOrder.SmallestDegree
    )

    with pytest.raises(AttributeError):
        gamma.evaluate_traces = False


@pytest.mark.parametrize(
    "operation",
    [
        "simplify_gamma",
        "collect_gamma_chains",
        "simplify_gamma0",
        "simplify_gamma_conjugate",
        "simplify_epsilon",
        "simplify_metrics",
        "simplify_color",
        "collect_color",
        "collect_color_constants",
        "to_cof_dimension_invariants",
        "canonize",
        "spenso_conjugate",
        "dirac_adjoint",
        "uncook",
        "schoonschip",
        "schoonschip_net",
        "to_dots",
        "normalize_dots",
        "expand_dots",
        "metric_shorthand_to_dot",
        "undo_all",
        "undo_schoonschip",
        "undo_dots",
        "undo_chain",
        "undo_trace",
        "normalize_chains",
        "undo_single_length",
    ],
)
def test_module_functions_drop_dispatch_while_methods_reinfer(operation: str):
    rep = Representation.euc(3)
    expression = TensorName.vector("idenso_parity_vector")(rep("mu"))

    module_result = getattr(idenso, operation)(expression)
    method_result = getattr(expression, operation)()

    assert isinstance(module_result, Expression)
    assert not isinstance(module_result, TensorExpression)
    assert isinstance(method_result, TensorExpression)
    assert method_result.to_expression() == module_result


def test_argument_operations_and_natural_result_types_are_paired():
    rep = Representation.euc(3)
    expression = TensorName.vector("idenso_parity_argument_vector", is_real=True)(
        rep("mu")
    )
    wrapper = S("index_wrapper")

    for operation, args in (
        ("wrap_indices", (wrapper,)),
        ("wrap_dummies", (wrapper,)),
        ("wrap_color", (S("color_wrapper"),)),
        ("collect_chains", (rep,)),
        ("chainify", (rep,)),
        ("conjugate_transpose", (rep,)),
    ):
        module_result = getattr(idenso, operation)(expression, *args)
        method_result = getattr(expression, operation)(*args)
        assert not isinstance(module_result, TensorExpression)
        assert isinstance(method_result, TensorExpression)
        assert method_result.to_expression() == module_result

    assert all(
        not isinstance(item, TensorExpression) for item in expression.list_dangling()
    )
    assert expression.list_dangling() == idenso.list_dangling(expression)

    root, aliases = expression.alias_subtensors("alias")
    assert (root, aliases) == idenso.alias_subtensors(expression, "alias")
    assert isinstance(root, Expression) and not isinstance(root, TensorExpression)
    assert all(
        not isinstance(item, TensorExpression) for pair in aliases for item in pair
    )

    patterns = [expression.to_expression()]
    assert expression.expand_in_patterns(patterns) == idenso.expand_in_patterns(
        expression, patterns
    )
    for operation in (
        "expand_mink",
        "expand_bis",
        "expand_mink_bis",
        "expand_metrics",
        "expand_color",
    ):
        terms = getattr(expression, operation)()
        assert terms == getattr(idenso, operation)(expression)
        assert all(
            not isinstance(item, TensorExpression) for pair in terms for item in pair
        )

    cooked = expression.cook()
    assert not isinstance(cooked, TensorExpression)
    assert cooked == idenso.cook(expression)

    cooked_function = expression.cook_function()
    assert not isinstance(cooked_function, TensorExpression)
    assert cooked_function == idenso.cook_function(expression)

    cooked_indices = expression.cook_indices()
    assert isinstance(cooked_indices, TensorExpression)
    assert cooked_indices.to_expression() == idenso.cook_indices(expression)


def test_targeted_exception_types_and_fallible_adapters():
    assert issubclass(idenso.CanonicalizationError, ValueError)
    assert issubclass(idenso.CookingError, TypeError)
    assert issubclass(idenso.DiracAdjointError, ValueError)
    assert issubclass(idenso.DotExpansionError, ValueError)
    assert issubclass(idenso.GammaConjugationError, ValueError)

    with pytest.raises(idenso.CookingError):
        idenso.cook_function(S("not_a_function"))

    bis = Representation.bis(4)
    three_open_spinors = TensorName("three_open_spinors", is_real=True)(
        bis("i"), bis("j"), bis("k")
    )
    with pytest.raises(idenso.DiracAdjointError):
        idenso.dirac_adjoint(three_open_spinors)
    with pytest.raises(idenso.DiracAdjointError):
        three_open_spinors.dirac_adjoint()


def test_representation_scalar_helpers_and_representation_aware_casimir_rewrite():
    d_f, d_a = S("dF"), S("dA")
    fundamental = Representation.cof(d_f)
    adjoint = Representation.coad(d_a)
    vector = TensorName.vector("idenso_color_vector")

    for invariant in (
        fundamental.dimension,
        fundamental.casimir(),
        fundamental.casimir(3),
        fundamental.dynkin_index(),
        fundamental.gram(2),
        fundamental.gram(3, adjoint),
    ):
        assert isinstance(invariant, Expression)
        assert not isinstance(invariant, TensorExpression)

    value = d_a * vector(adjoint)
    settings = idenso.ColorCasimirSettings(
        rewrite_fundamental_dimension=False,
        substitute_fundamental_index=False,
    )
    rewritten = value.to_color_casimir(
        fundamental=fundamental,
        adjoint=adjoint,
        settings=settings,
    )
    expected = (
        fundamental.dimension
        * fundamental.casimir()
        / fundamental.dynkin_index()
        * vector(adjoint)
    )
    assert rewritten.to_expression() == expected.to_expression()
    assert (
        idenso.to_color_casimir(
            value,
            fundamental=fundamental,
            adjoint=adjoint,
            settings=settings,
        )
        == rewritten.to_expression()
    )

    concrete_fundamental = Representation.cof(3)
    concrete_adjoint = Representation.coad(8)
    concrete = 8 * vector(concrete_adjoint)
    assert (
        concrete.to_color_casimir(
            fundamental=concrete_fundamental,
            adjoint=concrete_adjoint,
        ).to_expression()
        == concrete.to_expression()
    )

    ordinary_na = S("NA") * vector(adjoint)
    assert (
        ordinary_na.to_color_casimir(
            fundamental=fundamental,
            adjoint=adjoint,
            settings=settings,
        ).to_expression()
        == ordinary_na.to_expression()
    )


def test_untyped_color_structure_constants_are_rejected():
    a, b, c = S("a"), S("b"), S("c")
    structure_constant = TensorName.f()
    adjoint = Representation.coad(8)

    with pytest.raises(ValueError, match="three typed adjoint ports"):
        structure_constant(a, b, c)
    with pytest.raises(ValueError, match="three typed adjoint ports"):
        as_tensor(structure_constant.to_expression()(a, b, c))

    typed = structure_constant(adjoint("a"), adjoint("b"), adjoint("c"))
    assert isinstance(typed, TensorExpression)
    assert typed.rank == 3
