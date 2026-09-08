"""The DINOv2 cache across a resume, and the reproducibility question under it.

A three-hour Run all died twice with

    .../t4/dinov2_pool.npz describes rows ce0612f07b78 and this pool is 1685aa2c58a0

and the only cure was deleting the file by hand. These tests pin the repair and,
more importantly, pin the *distinction* the repair depends on:

**(A) the population legitimately differs between runs** — a different set of
candidate images was fetchable, so the detector ran again and its boxes are not
bit-reproducible. The old export is another population's file. Skipping it and
computing our own is correct, and nothing may be deleted.

**(B) the same inputs producing a different population** — that would be a
reproducibility bug in the selection pipeline, and automatic recomputation would
be hiding it. ``test_an_identical_export_gives_an_identical_population`` is the
test that would catch it, in-process and across a subprocess.

The protection itself is unchanged: a file is used only when its fingerprint
matches the rows being asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import proposals
from owl.active_selection import arms, population, semantic


def _rows(n=40, images=8):
    generator = np.random.default_rng(7)
    return (
        np.repeat(np.array([f"img{i:03d}" for i in range(images)]), n // images),
        np.clip(generator.random((n, 4)) * 0.5 + 0.25, 0.05, 0.95),
    )


@pytest.fixture
def embedder(monkeypatch):
    """Stand in for DINOv2, and count how often it is asked to work."""

    calls: list[int] = []

    def fake_embed(image_ids, boxes, jpeg_dir, **kwargs):
        calls.append(len(image_ids))
        block = np.random.default_rng(len(image_ids)).normal(size=(len(image_ids), 6))
        return (block / np.linalg.norm(block, axis=1, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(semantic, "embed", fake_embed)
    monkeypatch.setattr(semantic, "release", lambda **_: {"cuda": False})
    return calls


def _cached(path, image_ids, boxes, **kwargs):
    return semantic.cached(path, image_ids, boxes, "/nonexistent",
                           model_factory=lambda _device: object(),
                           device="cpu", label="test", **kwargs)


# ------------------------------------------------------------------ reuse ---


def test_a_valid_cache_is_reused_without_embedding_again(tmp_path, embedder):
    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"

    first = _cached(base, image_ids, boxes)
    assert len(embedder) == 1
    second = _cached(base, image_ids, boxes)
    assert len(embedder) == 1, "the second call embedded again"
    # fp16 on disk, so a hit equals the *rounded* computation, not the raw one.
    # See test_a_cache_hit_returns_fp16_rounded_features for why that matters
    # and why it is deliberately left alone.
    assert np.array_equal(second, first.astype(np.float16).astype(np.float32))

    written = sorted(p.name for p in tmp_path.glob("*.npz"))
    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    assert written == [semantic.cache_path(base, fingerprint).name]


def test_a_legacy_bare_name_is_still_reused_when_it_matches(tmp_path, embedder):
    """Exports written before the naming changed must not be recomputed."""

    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    semantic.write(base, np.zeros((len(image_ids), 6), dtype=np.float32),
                   fingerprint, {"legacy": True})

    features = _cached(base, image_ids, boxes)
    assert len(embedder) == 0, "a valid legacy export was recomputed"
    assert features.shape == (len(image_ids), 6)


# --------------------------------------------------- (A) a different pool ---


def test_another_populations_export_is_skipped_and_left_alone(tmp_path, embedder):
    """The exact failure that killed the run, and it must now be survivable.

    The old file is another population's. It is not read, it is **not deleted**,
    and the new population writes beside it.
    """

    old_ids, old_boxes = _rows()
    new_ids, new_boxes = _rows(n=40, images=8)
    new_boxes = new_boxes + 0.01                     # a different population
    base = tmp_path / "dinov2_pool.npz"

    _cached(base, old_ids, old_boxes)
    old_file = semantic.cache_path(base, semantic.row_fingerprint(old_ids, old_boxes))
    old_bytes = old_file.read_bytes()
    assert len(embedder) == 1

    features = _cached(base, new_ids, new_boxes)     # must not raise
    assert len(embedder) == 2
    assert features.shape[0] == len(new_ids)

    assert old_file.is_file(), "an unrelated population's export was deleted"
    assert old_file.read_bytes() == old_bytes, "it was overwritten"
    new_file = semantic.cache_path(base, semantic.row_fingerprint(new_ids, new_boxes))
    assert new_file.is_file() and new_file != old_file


def test_a_bare_legacy_export_for_another_population_is_not_fatal(tmp_path, embedder):
    """Reproduces the reported crash shape exactly: a *bare* stale file."""

    old_ids, old_boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    semantic.write(base, np.zeros((len(old_ids), 6), dtype=np.float32),
                   semantic.row_fingerprint(old_ids, old_boxes), {})

    new_ids, new_boxes = _rows()
    new_boxes = new_boxes + 0.02
    features = _cached(base, new_ids, new_boxes)     # used to raise SemanticError
    assert features.shape[0] == len(new_ids)
    assert base.is_file(), "the stale legacy export was deleted"


# ----------------------------------------------------------- corrupt file ---


@pytest.mark.parametrize("damage", ["truncate", "empty", "garbage"])
def test_a_corrupt_export_is_quarantined_and_recomputed(tmp_path, embedder, damage):
    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    _cached(base, image_ids, boxes)
    target = semantic.cache_path(base, semantic.row_fingerprint(image_ids, boxes))
    blob = target.read_bytes()
    target.write_bytes(
        b"" if damage == "empty"
        else b"not an npz" if damage == "garbage"
        else blob[: len(blob) // 2])

    features = _cached(base, image_ids, boxes)
    assert len(embedder) == 2
    assert features.shape[0] == len(image_ids)
    assert list(tmp_path.glob("*.unreadable")), "the bad file was not kept aside"


def test_an_unrelated_semantic_error_still_raises(tmp_path, embedder):
    """A version mismatch is a problem with the cache, not a different pool."""

    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    target = semantic.cache_path(base, fingerprint)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, features=np.zeros((len(image_ids), 6), dtype=np.float16),
        fingerprint=np.asarray(fingerprint),
        export_version=np.asarray("some_other_export_v9"),
        provenance=np.asarray("{}"))

    with pytest.raises(semantic.SemanticError) as raised:
        semantic.read(target, fingerprint=fingerprint)
    assert not isinstance(raised.value, semantic.SemanticFingerprintError)
    assert not isinstance(raised.value, semantic.SemanticCorruptError)


def test_the_fingerprint_check_is_still_enforced(tmp_path):
    """The guarantee this repair must not weaken."""

    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    fingerprint = semantic.row_fingerprint(image_ids, boxes)
    semantic.write(base, np.zeros((len(image_ids), 6), dtype=np.float32),
                   fingerprint, {})
    with pytest.raises(semantic.SemanticFingerprintError):
        semantic.read(base, fingerprint="0" * 64)


# --------------------------------------- (B) is the population reproducible ---


def _population_fingerprint(path):
    candidates = proposals.from_predict(path)
    pool = population.build(candidates)
    ranked = arms.ranked_positions("distribution_aware_iterative_v1", pool)
    return semantic.row_fingerprint(
        pool.candidates.image_ids[ranked], pool.candidates.boxes[ranked])


def _write_export(path, seed=0, images=24):
    generator = np.random.default_rng(seed)
    n = images * 20
    np.savez_compressed(
        path,
        image_ids=np.repeat(np.array([f"i{i:04d}" for i in range(images)]), 20),
        boxes=np.clip(generator.random((n, 4)) * 0.6 + 0.2, 0.05, 0.95).astype(np.float32),
        posterior=generator.dirichlet(np.full(81, 0.5), size=n).astype(np.float32),
        objectness=generator.random(n).astype(np.float32),
        embeddings=generator.normal(size=(n, 16)).astype(np.float32),
    )
    return path


def test_an_identical_export_gives_an_identical_population(tmp_path):
    """CASE B RULED OUT: the pipeline from a detector export to the rows the
    features describe is deterministic, in-process and in a fresh interpreter.

    If this ever fails, automatic recomputation would be **hiding a
    reproducibility bug** and the right response is to fix the pipeline, not the
    cache. That is why this test sits beside the cache tests.
    """

    import subprocess
    import sys
    from pathlib import Path

    export = _write_export(tmp_path / "proposals.npz")
    first = _population_fingerprint(export)
    assert _population_fingerprint(export) == first

    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(Path.cwd())!r});"
        "from tests.test_semantic_cache_resume import _population_fingerprint as f;"
        f"print(f({str(export)!r}))"
    )
    done = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == first, "the population is not reproducible"


def test_a_different_candidate_list_gives_a_different_population(tmp_path):
    """CASE A, and it is expected. This is what the stale cache was telling us."""

    a = _population_fingerprint(_write_export(tmp_path / "a.npz", seed=0))
    b = _population_fingerprint(_write_export(tmp_path / "b.npz", seed=1))
    assert a != b


def test_a_resume_reuses_valid_work_and_recomputes_only_the_stale_task(
    tmp_path, embedder
):
    """The end-to-end shape of the reported failure, in miniature.

    t2's population is unchanged, t3's is not. The resume must reuse t2 without
    embedding, compute t3, and destroy nothing.
    """

    ids2, boxes2 = _rows()
    ids3, boxes3 = _rows()
    boxes3 = boxes3 + 0.03
    t2, t3 = tmp_path / "t2", tmp_path / "t3"
    for directory in (t2, t3):
        directory.mkdir()

    _cached(t2 / "dinov2_pool.npz", ids2, boxes2)
    stale_ids, stale_boxes = _rows()
    stale_boxes = stale_boxes + 0.99
    _cached(t3 / "dinov2_pool.npz", stale_ids, stale_boxes)   # a dead run's t3
    assert len(embedder) == 2
    stale = semantic.cache_path(
        t3 / "dinov2_pool.npz", semantic.row_fingerprint(stale_ids, stale_boxes))
    assert stale.is_file()

    _cached(t2 / "dinov2_pool.npz", ids2, boxes2)             # resume: reuse
    assert len(embedder) == 2, "t2 was recomputed"
    _cached(t3 / "dinov2_pool.npz", ids3, boxes3)             # resume: recompute
    assert len(embedder) == 3

    assert stale.is_file(), "the dead run's export was deleted"
    assert len(list(t2.glob("*.npz"))) == 1
    assert len(list(t3.glob("*.npz"))) == 2


def test_a_cache_hit_returns_fp16_rounded_features(tmp_path, embedder):
    """A known, pre-existing wrinkle, pinned so it cannot surprise anyone.

    :func:`semantic.write` stores fp16, so a **cache hit** returns features
    rounded to fp16 while a **fresh computation** returns fp32. An uninterrupted
    run and a resumed one therefore differ in the last bits of the feature
    matrix, and a clusterer can in principle act on that.

    It is deliberately **not** changed here. Rounding the fresh path too would
    make fresh and resumed runs bit-identical, but it would also change what a
    re-run of the completed one-shot diagnostic and of the frozen benchmark's
    ``proposed``/``coreset``/``proposed_v2`` arms produces — a scientific
    behaviour change, which this repair is not allowed to make. Recorded instead.
    """

    image_ids, boxes = _rows()
    base = tmp_path / "dinov2_pool.npz"
    fresh = _cached(base, image_ids, boxes)
    hit = _cached(base, image_ids, boxes)

    assert fresh.dtype == hit.dtype == np.float32
    assert np.allclose(fresh, hit, atol=1e-3)
    assert np.array_equal(hit, fresh.astype(np.float16).astype(np.float32))
