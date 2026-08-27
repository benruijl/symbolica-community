"""Tests for symbolica-community extensions."""


def test_spenso_import():
    """Test that spenso module can be imported."""
    from symbolica.community.spenso import Representation, Tensor, TensorIndices

    assert Tensor is not None
    assert TensorIndices is not None
    assert Representation is not None


def test_feynkit_import():
    """FeynKit is registered in the shared Symbolica extension."""
    from symbolica.community.feynkit import (
        FourMomentum,
        Generator,
        Model,
        ParticleSelector,
    )

    assert FourMomentum is not None
    assert Generator is not None
    assert Model is not None
    assert ParticleSelector is not None
