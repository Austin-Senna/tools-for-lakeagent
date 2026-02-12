"""Tests for aurum.config."""

from aurum.config import aurumConfig


def test_defaults():
    cfg = aurumConfig()
    assert cfg.minhash_perms == 256
    assert cfg.jaccard_threshold == 0.7
    assert cfg.pk_cardinality_threshold == 0.7
    assert cfg.max_hops == 3


def test_immutable():
    cfg = aurumConfig()
    try:
        cfg.minhash_perms = 512  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_custom_values():
    cfg = aurumConfig(minhash_perms=128, jaccard_threshold=0.5)
    assert cfg.minhash_perms == 128
    assert cfg.jaccard_threshold == 0.5
    # Ensure other defaults are unchanged
    assert cfg.max_hops == 3
