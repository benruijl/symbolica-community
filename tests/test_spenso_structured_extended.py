"""Extended integration checks for the structured Spenso public API."""

import shutil
import subprocess

import pytest
from symbolica import S, T
from symbolica.community.spenso import (
    _,
    BroadcastFunction,
    Representation,
    Slot,
    Tensor,
    TensorExpression,
    TensorFunctionLibrary,
    TensorLibrary,
    TensorName,
    TensorNetwork,
    as_tensor,
    chain,
    dot,
    format_tensor,
    formatted,
    to_typst,
    trace,
)


BOOLEAN_SYMBOL_OPTIONS = (
    "is_symmetric",
    "is_antisymmetric",
    "is_cyclesymmetric",
    "is_linear",
    "is_flat",
    "is_scalar",
    "is_real",
    "is_integer",
    "is_positive",
)
CALLBACK_SYMBOL_OPTIONS = (
    "normalization",
    "print",
    "derivative",
    "series",
    "eval",
    "data",
)


def option_value(option: str, prefix: str):
    if option in BOOLEAN_SYMBOL_OPTIONS:
        return True
    if option == "tags":
        return [f"{prefix}_tag"]
    if option == "aliases":
        return [f"{prefix}_alias"]
    if option == "normalization":
        return T()
    if option == "print":
        return lambda *_args, **_kwargs: None
    if option == "derivative":
        return lambda function, _index: function
    if option == "series":
        return lambda _args: None
    if option == "eval":
        return {
            "float": lambda args: args[0],
            "complex": lambda args: args[0],
        }
    if option == "data":
        return {"source": prefix}
    raise AssertionError(f"unhandled constructor option {option}")


@pytest.mark.parametrize(
    "option", BOOLEAN_SYMBOL_OPTIONS + ("tags", "aliases") + CALLBACK_SYMBOL_OPTIONS
)
def test_tensor_name_forwards_every_symbol_constructor_option(option):
    prefix = f"structured_tensor_option_{option}"
    name = TensorName(prefix, **{option: option_value(option, prefix)})

    assert name.has_tag("spenso::tensor")
    if option == "tags":
        assert name.has_tag(f"{prefix}_tag")


@pytest.mark.parametrize(
    "option", BOOLEAN_SYMBOL_OPTIONS + ("tags", "aliases") + CALLBACK_SYMBOL_OPTIONS
)
def test_broadcast_forwards_every_symbol_constructor_option(option):
    prefix = f"structured_broadcast_option_{option}"
    function = BroadcastFunction(prefix, **{option: option_value(option, prefix)})

    assert function.has_tag("spenso::broadcast")
    assert not function.has_tag("spenso::tensor")
    if option == "tags":
        assert function.has_tag(f"{prefix}_tag")


def test_tensor_rank_options_and_symbol_callback_parity():
    TensorName("structured_rank_none", rank=None)
    rank_one = TensorName("structured_rank_one", rank=1)
    assert rank_one.has_tag("spenso::rank1")

    def tensor_printer(*_args, **_kwargs):
        return "tensor_callback_print"

    def broadcast_printer(*_args, **_kwargs):
        return "broadcast_callback_print"

    tensor_print = TensorName("structured_tensor_print", print=tensor_printer)
    broadcast_print = BroadcastFunction(
        "structured_broadcast_print", print=broadcast_printer
    )
    assert str(tensor_print.to_expression()) == "tensor_callback_print"
    assert str(broadcast_print.to_expression()) == "broadcast_callback_print"

    tensor_data = TensorName("structured_tensor_data", data={"origin": "tensor"})
    broadcast_data = BroadcastFunction(
        "structured_broadcast_data", data={"origin": "broadcast"}
    )
    assert tensor_data.to_expression().get_symbol_data("origin") == "tensor"
    assert broadcast_data.to_expression().get_symbol_data("origin") == "broadcast"

    x = S("structured_callback_x")
    tensor_marker = S("structured_tensor_derivative_marker")
    broadcast_marker = S("structured_broadcast_derivative_marker")
    mink = Representation.mink(4)
    tensor_derivative = TensorName(
        "structured_tensor_derivative",
        derivative=lambda _function, _index: tensor_marker,
    )
    broadcast_derivative = BroadcastFunction(
        "structured_broadcast_derivative",
        derivative=lambda _function, _index: broadcast_marker,
    )

    tensor_value = as_tensor(tensor_derivative(x, mink)).to_expression()
    assert tensor_value.derivative(x) == tensor_marker
    assert broadcast_derivative(x).derivative(x) == broadcast_marker


def test_malformed_python_and_parser_broadcast_arities_fail():
    function = BroadcastFunction("structured_malformed_broadcast")
    x = S("structured_malformed_x")
    y = S("structured_malformed_y")

    with pytest.raises(TypeError):
        function()
    with pytest.raises(TypeError):
        function(x, y)

    head = function.to_expression()
    for malformed in (head(), head(x, y)):
        with pytest.raises(RuntimeError, match="Too many arguments for function"):
            TensorNetwork(malformed)


def test_module_display_formats_a_base_expression_without_mutation():
    mink = Representation.mink(4)
    p = TensorName.vector("structured_display_p")
    q = TensorName.vector("structured_display_q")
    expression = dot(p(mink), q(mink)).to_expression()
    canonical = expression.to_canonical_string()

    assert not isinstance(expression, TensorExpression)
    assert isinstance(format_tensor(expression), str)
    assert isinstance(to_typst(expression), str)
    assert str(formatted(expression)) == format_tensor(expression)
    assert "4" not in format_tensor(expression)
    assert "4" in format_tensor(expression, show_dimensions=True)
    assert expression.to_canonical_string() == canonical


def test_typst_display_examples_compile_to_svg_without_mutation(tmp_path):
    typst = shutil.which("typst")
    if typst is None:
        pytest.skip("Typst compiler is not installed")

    euc = Representation.euc(2)
    mink = Representation.mink(2)
    bis = Representation.bis(2)
    p = TensorName.vector("structured_typst_p")
    q = TensorName.vector("structured_typst_q")
    gamma = TensorName.gamma()(mink, bis, bis)
    factors = (gamma(mink("mu"), _, _), gamma(mink("nu"), _, _))
    outer = p(euc("i")).outer(q(mink("j")))
    user_defined = TensorName("structured_typst_user")(
        mink("logical_m"), bis("logical_b")
    )
    concrete = Tensor.dense(
        outer.with_name(TensorName("structured_typst_dense")),
        [1.0, 2.0, 3.0, 4.0],
    )
    expressions = [
        p(euc("i")),
        gamma(mink("mu"), bis("a"), bis("b")),
        dot(p(euc), q(euc)),
        chain(bis("a"), bis("b"), *factors),
        trace(bis, *factors),
        outer,
        user_defined,
    ]
    for rendered in (
        user_defined.format_tensor(),
        user_defined.to_typst(),
        user_defined._repr_latex_(),
    ):
        assert rendered.find("logical_m") < rendered.find("logical_b")
    canonical = [value.to_expression().to_canonical_string() for value in expressions]
    sources = [value.to_typst() for value in expressions]
    sources.append(to_typst(expressions[2].to_expression()))
    sources.append(concrete.to_typst())

    for index, source in enumerate(sources):
        subprocess.run(
            [typst, "compile", "--format", "svg", "-", tmp_path / f"{index}.svg"],
            input=f"$ {source} $\n",
            text=True,
            check=True,
        )

    assert [
        value.to_expression().to_canonical_string() for value in expressions
    ] == canonical


def test_explicit_dot_chain_and_trace_helpers_preserve_spectator_order():
    mink = Representation.mink(4)
    bis = Representation.bis(4)
    mu = mink("mu")
    nu = mink("nu")
    p = TensorName.vector("structured_helper_p")
    q = TensorName.vector("structured_helper_q")
    gamma = TensorName.gamma()(mink, bis, bis)

    dotted = dot(p(mink), q(mink))
    assert dotted.is_scalar
    assert "dot" in dotted.to_expression().to_canonical_string()

    factors = (gamma.index(mu, _, _), gamma.index(nu, _, _))
    ordered = chain(bis("i"), bis("j"), *factors)
    assert "chain" in ordered.to_expression().to_canonical_string()
    assert [str(port) for port in ordered.interface if isinstance(port, Slot)][-2:] == [
        str(mu),
        str(nu),
    ]

    closed = trace(bis, *factors)
    assert "trace" in closed.to_expression().to_canonical_string()
    assert [str(port) for port in closed.interface] == [str(mu), str(nu)]


def test_named_composite_descriptors_preserve_logical_layout_and_library_provenance():
    euc = Representation.euc(2)
    mink = Representation.mink(3)
    p = TensorName.vector("structured_composite_p")
    q = TensorName.vector("structured_composite_q")

    unnamed_explicit = p(euc("structured_composite_i")).outer(
        q(mink("structured_composite_j"))
    )
    with pytest.raises(ValueError, match="no name"):
        Tensor.dense(unnamed_explicit, list(range(6)))

    explicit = unnamed_explicit.with_name(TensorName("structured_composite_data"))
    tensor = Tensor.dense(explicit, list(range(6)))
    assert explicit[4] == [1, 1]
    assert explicit[[1, 1]] == 4
    assert tensor[[1, 2]] == pytest.approx(5.0)
    assert isinstance(tensor.structure(), TensorExpression)

    unresolved = p(euc).outer(q(mink))
    atomic_open = Tensor.dense(p(euc), list(range(2)))
    assert atomic_open.structure().interface == (euc,)
    with pytest.raises(ValueError, match="no name"):
        Tensor.dense(unresolved, list(range(6)))

    definition = unresolved.with_name(TensorName("structured_composite_library"))
    explicit_again = Tensor.dense(explicit, list(range(6)))
    assert explicit_again.structure().interface == explicit.interface
    indexed_definition = definition("structured_composite_u", "structured_composite_v")
    assert str(indexed_definition.name) == str(definition.name)
    assert (definition * 1).name is None
    stored = Tensor.dense(definition, list(range(6)))
    library = TensorLibrary()
    with pytest.raises(ValueError, match="fully unresolved"):
        library.register(explicit_again)
    library.register(stored)

    composite = stored.structure()
    atomic = library[definition.name]
    atomic_by_string = library["structured_composite_library"]
    assert isinstance(composite, TensorExpression)
    assert isinstance(atomic, TensorExpression)
    assert composite.to_expression() == definition.to_expression()
    assert atomic.to_expression() != composite.to_expression()
    assert atomic.interface == composite.interface
    assert atomic_by_string.to_expression() == atomic.to_expression()

    network = TensorNetwork(
        atomic("structured_composite_u", "structured_composite_v"), library=library
    )
    network.execute(library=library)
    result = network.result_tensor(library=library)
    assert result[:] == pytest.approx(list(range(6)))
    assert result[[1, 2]] == pytest.approx(5.0)


def test_scalar_library_reference_materializes_as_named_rank_zero_tensor():
    representation = Representation.euc(2)
    vector = TensorName.vector("structured_scalar_library_vector")
    definition = (as_tensor(vector(representation)) * vector(representation)).with_name(
        TensorName("structured_scalar_library")
    )
    stored = Tensor.dense(definition, [7.0])
    library = TensorLibrary()
    library.register(stored)

    atomic = library["structured_scalar_library"]
    assert atomic.is_scalar
    network = TensorNetwork(atomic, library=library)
    network.execute(library=library)
    result = network.result_tensor(library=library)

    assert str(result.scalar()).startswith("7")
    assert result[:] == pytest.approx([7.0])
    assert result.structure().is_scalar
    assert str(result.structure().name) == str(definition.name)


def test_library_atomic_lookup_preserves_scalar_key_arguments():
    representation = Representation.euc(2)
    metadata = S("structured_library_metadata")
    head = TensorName.vector("structured_library_with_metadata")
    definition = as_tensor(head(metadata, representation))
    library = TensorLibrary()
    library.register(Tensor.dense(definition, [2.0, 3.0]))

    atomic = library[head]
    assert "structured_library_metadata" in atomic.to_expression().to_canonical_string()
    network = TensorNetwork(atomic("structured_library_metadata_i"), library=library)
    network.execute(library=library)
    result = network.result_tensor(library=library)

    assert result[:] == pytest.approx([2.0, 3.0])
    assert "structured_library_metadata" in (
        result.structure().to_expression().to_canonical_string()
    )


def test_sparse_data_validates_dimensions_and_returns_implicit_zeros():
    representation = Representation.euc(3)
    vector = TensorName.vector("structured_sparse_zero")
    sparse = Tensor.sparse(vector(representation), float)

    assert sparse[:] == pytest.approx([0.0, 0.0, 0.0])
    assert sparse[[1]] == pytest.approx(0.0)
    sparse[2] = 9.0
    assert sparse[:] == pytest.approx([0.0, 0.0, 9.0])

    symbolic = Representation("structured_symbolic_dimension", S("structured_D"))
    with pytest.raises(ValueError, match="concrete representation dimensions"):
        Tensor.sparse(
            TensorName.vector("structured_symbolic_sparse_tensor")(
                symbolic("structured_i")
            ),
            float,
        )
    with pytest.raises(ValueError, match="concrete representation dimensions"):
        Tensor.sparse(
            as_tensor(
                TensorName.vector("structured_symbolic_sparse_library")(symbolic)
            ),
            float,
        )


def test_evaluator_preserves_named_descriptor_and_logical_data_order():
    parameter = S("structured_evaluator_parameter")
    mink = Representation.mink(2)
    euc = Representation.euc(3)
    descriptor = TensorName("structured_evaluator_source")(
        mink("structured_evaluator_m"), euc("structured_evaluator_e")
    )
    tensor = Tensor.dense(descriptor, [parameter + value for value in range(6)])
    evaluator = tensor.evaluator(constants={}, funs={}, params=[parameter])

    result = evaluator.evaluate_complex([[10.0 + 0.0j]])[0]

    assert result[:] == pytest.approx([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert result[[1, 2]] == pytest.approx(15.0)
    assert result.structure().interface == descriptor.interface
    assert str(result.structure().name) == str(descriptor.name)


def test_registered_broadcast_callback_executes_on_a_concrete_tensor():
    representation = Representation.euc(2)
    vector = TensorName.vector("structured_concrete_vector")
    sqrt = BroadcastFunction("structured_concrete_sqrt", is_real=True)
    tensor_library = TensorLibrary()
    vector_expression = as_tensor(vector(representation))
    tensor_library.register(Tensor.dense(vector_expression, [1.0, 4.0]))
    expression = sqrt(vector_expression).with_name(
        TensorName("structured_concrete_sqrt_result")
    )

    missing = TensorNetwork(expression, library=tensor_library)
    with pytest.raises(
        RuntimeError,
        match="no concrete callback registered for broadcast function",
    ):
        missing.execute(
            library=tensor_library,
            function_library=TensorFunctionLibrary(),
        )

    observed = []
    functions = TensorFunctionLibrary()

    def callback(value):
        observed.append(value)
        return value**0.5

    functions.register(sqrt, callback)
    network = TensorNetwork(expression, library=tensor_library)
    network.execute(library=tensor_library, function_library=functions)
    result = network.result_tensor(library=tensor_library)

    assert observed == [1.0, 4.0]
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(2.0)


def test_registered_broadcast_callback_executes_on_a_complex_tensor():
    representation = Representation.euc(2)
    vector = TensorName.vector("structured_complex_vector")
    conjugate = BroadcastFunction("structured_complex_conjugate")
    tensor_library = TensorLibrary()
    vector_expression = as_tensor(vector(representation))
    tensor_library.register(Tensor.dense(vector_expression, [1.0 + 2.0j, 3.0 - 4.0j]))
    expression = conjugate(vector_expression).with_name(
        TensorName("structured_complex_conjugate_result")
    )

    functions = TensorFunctionLibrary()
    functions.register(conjugate, lambda value: value.conjugate())
    network = TensorNetwork(expression, library=tensor_library)
    network.execute(library=tensor_library, function_library=functions)
    result = network.result_tensor(library=tensor_library)

    assert result[:] == pytest.approx([1.0 - 2.0j, 3.0 + 4.0j])
