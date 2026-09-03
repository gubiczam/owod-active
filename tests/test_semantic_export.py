"""The live DINOv2 export's contract, without a GPU or a backbone download.

The GPU half of :mod:`owl.active_selection.semantic` cannot be tested on a
laptop. Everything around it can, and one thing here must be: the **cache key**.

Benchmark V1 runs one semantic pass per coverage arm per task and caches it
beside that task's proposals. Two arms in the same chain embed *different* row
sets — the gated arm embeds only the admissible subset — and three tasks embed
different populations entirely. A cache that answered with the wrong file would
select using one population's geometry over another's, and nothing downstream
would say so. So the file carries a fingerprint of exactly the rows it
describes, and reading it with a different population is an error rather than a
silent success.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import semantic_features as sf
from owl.active_selection import semantic


@pytest.fixture
def rows():
    image_ids = np.asarray(["000000000001", "000000000001", "000000000002"])
    boxes = np.asarray([
        [0.5, 0.5, 0.2, 0.2],
        [0.25, 0.25, 0.1, 0.1],
        [0.75, 0.75, 0.3, 0.4],
    ], dtype=np.float32)
    return image_ids, boxes


def unit(n: int, dim: int = sf.FEATURE_DIM, seed: int = 0) -> np.ndarray:
    block = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------ fingerprint ---


def test_the_fingerprint_is_the_identity_of_the_rows(rows):
    image_ids, boxes = rows
    assert semantic.row_fingerprint(image_ids, boxes) == semantic.row_fingerprint(
        image_ids, boxes
    )


def test_a_different_image_changes_the_fingerprint(rows):
    image_ids, boxes = rows
    other = image_ids.copy()
    other[0] = "000000000009"
    assert semantic.row_fingerprint(other, boxes) != semantic.row_fingerprint(
        image_ids, boxes
    )


def test_a_different_box_changes_the_fingerprint(rows):
    image_ids, boxes = rows
    moved = boxes.copy()
    moved[1, 0] += 0.05
    assert semantic.row_fingerprint(image_ids, moved) != semantic.row_fingerprint(
        image_ids, boxes
    )


def test_a_subset_of_the_rows_changes_the_fingerprint(rows):
    """The gated arm embeds a subset, and must not reuse the ungated cache."""

    image_ids, boxes = rows
    assert semantic.row_fingerprint(
        image_ids[:2], boxes[:2]
    ) != semantic.row_fingerprint(image_ids, boxes)


def test_reordering_the_rows_changes_the_fingerprint(rows):
    """Features are consumed positionally, so row order is part of the identity."""

    image_ids, boxes = rows
    order = np.asarray([2, 0, 1])
    assert semantic.row_fingerprint(
        image_ids[order], boxes[order]
    ) != semantic.row_fingerprint(image_ids, boxes)


def test_a_float32_round_trip_does_not_change_the_fingerprint(rows):
    """PROB writes float32; a key that moved with the last bit would be useless."""

    image_ids, boxes = rows
    reread = boxes.astype(np.float32).astype(np.float64).astype(np.float32)
    assert semantic.row_fingerprint(image_ids, reread) == semantic.row_fingerprint(
        image_ids, boxes
    )


# ----------------------------------------------------------------- the file ---


def test_write_then_read_returns_the_features(tmp_path, rows):
    image_ids, boxes = rows
    features = unit(3)
    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    path = semantic.write(tmp_path / "f.npz", features, fingerprint, {"task": "t2"})
    back = semantic.read(path, fingerprint=fingerprint)
    assert back.shape == features.shape
    # fp16 on disk, which is what every committed export in this project uses
    assert np.allclose(back, features, atol=1e-3)


def test_the_write_is_atomic(tmp_path, rows):
    semantic.write(tmp_path / "deep" / "f.npz", unit(3),
                   semantic.row_fingerprint(*rows), {})
    assert (tmp_path / "deep" / "f.npz").is_file()
    assert not list(tmp_path.rglob("*.part"))


def test_reading_with_another_populations_fingerprint_is_refused(tmp_path, rows):
    """The check that stops one task's geometry deciding another's selection."""

    image_ids, boxes = rows
    path = semantic.write(tmp_path / "f.npz", unit(3),
                          semantic.row_fingerprint(image_ids, boxes), {})
    with pytest.raises(semantic.SemanticError, match="describes rows"):
        semantic.read(path, fingerprint="0" * 64)


def test_reading_an_export_of_another_version_is_refused(tmp_path):
    path = tmp_path / "f.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, features=unit(3).astype(np.float16),
            fingerprint=np.asarray("x" * 64),
            export_version=np.asarray("some_older_export_v0"),
            provenance=np.asarray("{}"),
        )
    with pytest.raises(semantic.SemanticError, match="this code reads"):
        semantic.read(path)


def test_the_cache_is_used_when_the_rows_match(tmp_path, rows):
    """A resumed session must not pay for the pass twice."""

    image_ids, boxes = rows
    calls: list[int] = []

    def backbone(_device):
        calls.append(1)
        raise AssertionError("the backbone must not be loaded for a cache hit")

    semantic.write(tmp_path / "f.npz", unit(3),
                   semantic.row_fingerprint(image_ids, boxes), {})
    got = semantic.cached(
        tmp_path / "f.npz", image_ids, boxes, tmp_path,
        model_factory=backbone, device="cpu",
    )
    assert got.shape == (3, sf.FEATURE_DIM)
    assert not calls


def test_the_cache_is_refused_rather_than_reused_for_other_rows(tmp_path, rows):
    image_ids, boxes = rows
    semantic.write(tmp_path / "f.npz", unit(3),
                   semantic.row_fingerprint(image_ids, boxes), {})
    moved = boxes.copy()
    moved[0, 0] += 0.1
    with pytest.raises(semantic.SemanticError, match="delete the file"):
        semantic.cached(
            tmp_path / "f.npz", image_ids, moved, tmp_path,
            model_factory=lambda _d: None, device="cpu",
        )


# --------------------------------------------------------------- the crop ---


def test_the_crop_is_the_frozen_method_v2_one():
    """Imported, not restated: the two paths must move together or not at all."""

    import inspect

    source = inspect.getsource(semantic)
    assert "sf.square_crop(" in source
    assert "sf.CROP_SIZE" in source
    assert f"{sf.CROP_MARGIN}" not in source, (
        "the margin must be read from owl.semantic_features, not restated"
    )
    assert sf.crop_specification()["margin"] == sf.CROP_MARGIN


def test_the_reference_rows_are_unit_norm(tmp_path, monkeypatch):
    """REF-T1 is reused as-is; every cosine downstream assumes unit norm."""

    block = np.random.default_rng(0).normal(size=(5, sf.FEATURE_DIM))
    monkeypatch.setitem(
        __import__("sys").modules, "tools.export_ref_t1_features",
        type("M", (), {"read": staticmethod(lambda _p: {"embeddings": block})})(),
    )
    out = semantic.reference_from_ref_t1("ignored")
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_stacking_drops_empty_blocks():
    stacked = semantic.stack_reference([
        None,
        np.zeros((0, 8), dtype=np.float32),
        unit(2, dim=8),
        unit(3, dim=8, seed=1),
    ])
    assert stacked.shape == (5, 8)


def test_stacking_nothing_gives_a_well_shaped_empty_reference():
    stacked = semantic.stack_reference([])
    assert stacked.shape == (0, sf.FEATURE_DIM)
