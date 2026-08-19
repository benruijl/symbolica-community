import pytest

import symbolica.community.spenso as spenso
from symbolica import Expression, S
from symbolica.community.spenso import (
    _,
    BroadcastFunction,
    Representation,
    Tensor,
    TensorExpression,
    TensorFunctionLibrary,
    TensorLibrary,
    TensorName,
    TensorNetwork,
    dot,
)


def test_only_one_public_symbolic_structure_and_data_tensor_type():
    assert not hasattr(spenso, "TensorStructure")
    assert not hasattr(spenso, "TensorIndices")
    assert not hasattr(spenso, "LibraryTensor")
    assert not hasattr(spenso, "TensorNamespace")


def test_tensor_name_always_constructs_a_tensor_expression():
    euc = Representation.euc(2)
    tensor = TensorName("unified_constructor_tensor")
    vector = TensorName.vector("unified_constructor_vector")

    rank_zero = tensor(7)
    unresolved = tensor(7, euc, euc)
    mixed = tensor(7, euc("mu"), euc)
    explicit = tensor(7, euc("mu"), euc("nu"))

    assert all(
        isinstance(value, TensorExpression)
        for value in (rank_zero, unresolved, mixed, explicit)
    )
    assert rank_zero.is_scalar
    assert unresolved.interface == (euc, euc)
    assert mixed.interface == (euc("mu"), euc)
    assert explicit.interface == (euc("mu"), euc("nu"))
    assert isinstance(vector(1, euc), TensorExpression)

    with pytest.raises(ValueError, match="exactly one"):
        vector(1)
    with pytest.raises(ValueError, match="precede"):
        tensor(euc, 1)


def test_tensor_expression_indexing_and_expand_preserve_structure():
    euc = Representation.euc(2)
    x = S("unified_expand_x")
    vector = TensorName.vector("unified_expand_vector")
    value = ((1 + x) * vector(3, euc)).with_name(TensorName("unified_expand_result")(9))

    indexed = value.index(euc("mu"))
    assert indexed.interface == (euc("mu"),)

    expanded = value.expand()
    assert isinstance(expanded, TensorExpression)
    assert expanded.interface == value.interface
    assert str(expanded.name) == str(value.name)
    assert not isinstance(value.to_expression().expand(), TensorExpression)


def test_tensor_data_accepts_unresolved_explicit_and_mixed_interfaces():
    euc = Representation.euc(2)
    tensor = TensorName("unified_data_tensor")
    concrete_values = []

    for descriptor in (
        tensor(0, euc, euc),
        tensor(1, euc("mu"), euc),
        tensor(2, euc("mu"), euc("nu")),
    ):
        concrete = Tensor.dense(descriptor, [1.0, 2.0, 3.0, 4.0])
        concrete_values.append(concrete)
        assert concrete.structure().interface == descriptor.interface
        assert list(concrete) == [1.0, 2.0, 3.0, 4.0]

        sparse = Tensor.sparse(descriptor, float)
        for index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            sparse[index] = value
        assert sparse.structure().interface == descriptor.interface
        assert list(sparse) == [1.0, 2.0, 3.0, 4.0]

    mixed_reference = concrete_values[1].index(euc("nu"))
    assert isinstance(mixed_reference, TensorNetwork)
    with pytest.raises(ValueError, match="expected 1"):
        concrete_values[1].index(euc("nu"), euc("rho"))


def test_library_uses_full_atomic_key_and_registers_tensor():
    euc = Representation.euc(2)
    vector = TensorName.vector("unified_library_vector")
    first = vector(1, euc)
    second = vector(2, euc)
    library = TensorLibrary()

    library.register(Tensor.dense(first, [1.0, 2.0]))
    library.register(Tensor.dense(second, [3.0, 4.0]))

    assert library[first].interface == first.interface
    assert library[second].interface == second.interface
    with pytest.raises((KeyError, ValueError), match="ambiguous|signatures"):
        library[vector]


def test_concrete_arithmetic_and_helpers_promote_to_network():
    euc = Representation.euc(2)
    left = Tensor.dense(TensorName.vector("unified_network_left")(euc), [1.0, 2.0])
    right = Tensor.dense(TensorName.vector("unified_network_right")(euc), [3.0, 4.0])

    inferred = left * right
    explicit = dot(left, right)
    assert isinstance(inferred, TensorNetwork)
    assert isinstance(explicit, TensorNetwork)

    inferred.execute()
    result = inferred.result_tensor()
    assert result.structure().is_scalar
    assert result.structure().name is None
    assert result[0] == pytest.approx(11.0)


def test_concrete_private_indices_do_not_contract_unequal_explicit_ports():
    euc = Representation.euc(2)
    left = Tensor.dense(
        TensorName.vector("unified_explicit_left")(euc("left_i")), [1.0, 2.0]
    )
    right = Tensor.dense(
        TensorName.vector("unified_explicit_right")(euc("right_i")), [3.0, 4.0]
    )

    network = left * right
    assert network.structure().rank == 2
    network.execute()
    assert list(network.result_tensor()) == pytest.approx([3.0, 4.0, 6.0, 8.0])


def test_concrete_matrix_products_infer_and_flatten_chains():
    euc = Representation.euc(2)
    matrix = TensorName("unified_matrix")
    left = Tensor.dense(matrix(1, euc, euc), [1.0, 2.0, 3.0, 4.0])
    right = Tensor.dense(matrix(2, euc, euc), [5.0, 6.0, 7.0, 8.0])
    diagonal = Tensor.dense(matrix(3, euc, euc), [2.0, 0.0, 0.0, 3.0])
    shear = Tensor.dense(matrix(4, euc, euc), [1.0, 1.0, 0.0, 1.0])

    product = left * right
    extended = product * left
    prepended = left * (right * left)
    joined = (left * right) * (diagonal * shear)
    assert isinstance(product, TensorNetwork)
    assert product.structure().rank == 2
    for network in (product, extended, prepended, joined):
        assert (
            network.structure().to_expression().to_canonical_string().count("chain(")
            == 1
        )

    product.execute()
    extended.execute()
    prepended.execute()
    joined.execute()
    assert list(product.result_tensor()) == pytest.approx([19.0, 22.0, 43.0, 50.0])
    assert list(extended.result_tensor()) == pytest.approx([85.0, 126.0, 193.0, 286.0])
    assert list(prepended.result_tensor()) == pytest.approx([85.0, 126.0, 193.0, 286.0])
    assert list(joined.result_tensor()) == pytest.approx([38.0, 104.0, 86.0, 236.0])


def test_to_network_and_concrete_broadcast_use_the_public_callbacks():
    euc = Representation.euc(2)
    vector = TensorName.vector("unified_broadcast_vector")
    descriptor = vector(euc)
    concrete = Tensor.dense(descriptor, [1.0, 4.0])

    symbolic = (vector(euc) * 0).to_network()
    assert isinstance(symbolic, TensorNetwork)

    sqrt = BroadcastFunction("unified_broadcast_sqrt", is_real=True)
    functions = TensorFunctionLibrary()
    functions.register(sqrt, lambda value: value**0.5)

    network = sqrt(concrete)
    assert isinstance(network, TensorNetwork)
    network.execute(function_library=functions)
    assert list(network.result_tensor()) == [1.0, 2.0]


def test_composite_parameterized_name_is_metadata_not_atom_rewrite():
    euc = Representation.euc(2)
    left = TensorName.vector("unified_named_left")(euc)
    right = TensorName.vector("unified_named_right")(euc)
    name = TensorName("unified_named_composite")
    composite = left.outer(right)
    definition = composite.with_name(name(1))

    assert definition.to_expression() == composite.to_expression()
    concrete = Tensor.dense(definition, [1.0, 0.0, 0.0, 1.0])
    assert concrete.structure().to_expression() == composite.to_expression()

    library = TensorLibrary()
    library.register(concrete)
    exact = library[name(1, euc, euc)]
    assert exact.interface == definition.interface
    assert "1" in exact.to_expression().to_canonical_string()

    second = concrete.with_name(name(2))
    library.register(second)
    assert "2" in library[name(2, euc, euc)].to_expression().to_canonical_string()
    with pytest.raises(KeyError, match="registered signatures"):
        library[name]


def test_broadcast_scalar_remains_an_ordinary_expression():
    x = S("unified_broadcast_scalar")
    function = BroadcastFunction("unified_scalar_function")
    result = function(x)

    assert isinstance(result, Expression)
    assert not isinstance(result, TensorExpression)
