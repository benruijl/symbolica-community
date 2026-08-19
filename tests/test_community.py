"""Tests for symbolica-community extensions."""
import pytest


def test_spenso_import():
    """Test that spenso module can be imported."""
    from symbolica.community.spenso import (
        Representation,
        Tensor,
        TensorExpression,
        TensorName,
    )

    assert Tensor is not None
    assert TensorExpression is not None
    assert TensorName is not None
    assert Representation is not None
