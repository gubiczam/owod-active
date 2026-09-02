"""Method V2's export contract: crop geometry, pool-only filtering, and the gates.

The crop tests carry most of the weight. A padded crop would inject a constant
synthetic region whose *area varies with how close a proposal sits to the image
edge* -- an edge-proximity signal written straight into the embedding, and
indistinguishable from semantics in every metric the audit computes. So "square,
inside the image, never padded" is a property worth asserting rather than
assuming, especially at the boundaries where it is easiest to get wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import semantic_features as sf
from owl.decoder_layers import ExportError, proposal_keys

# ------------------------------------------------------------ crop geometry ---


@pytest.mark.parametrize(
    ("cx", "cy", "w", "h", "width", "height"),
    [
        (0.5, 0.5, 0.10, 0.05, 640, 480),     # comfortably interior
        (0.01, 0.01, 0.10, 0.05, 640, 480),   # top-left corner
        (0.99, 0.99, 0.10, 0.05, 640, 480),   # bottom-right corner
        (0.5, 0.02, 0.30, 0.05, 640, 480),    # wide box against the top edge
        (0.02, 0.5, 0.05, 0.30, 640, 480),    # tall box against the left edge
        (0.5, 0.5, 2.00, 2.00, 640, 480),     # larger than the image
        (0.5, 0.5, 0.001, 0.001, 640, 480),   # degenerately small
        (0.5, 0.5, 0.20, 0.20, 17, 23),       # tiny, odd-sized image
    ],
)
def test_crop_is_square_inside_the_image_and_non_empty(cx, cy, w, h, width, height):
    x0, y0, x1, y1 = sf.square_crop(cx, cy, w, h, width, height)

    assert (x1 - x0) == (y1 - y0), "a non-square crop would be resized anisotropically"
    assert x1 > x0 and y1 > y0, "an empty crop cannot be embedded"
    assert 0 <= x0 and x1 <= width, "crop leaves the image horizontally"
    assert 0 <= y0 and y1 <= height, "crop leaves the image vertically"


def test_crop_side_is_the_margin_times_the_larger_dimension():
    """The frozen rule: 1.20x the *larger* proposal side, not the diagonal or area."""

    x0, _y0, x1, _y1 = sf.square_crop(0.5, 0.5, 0.10, 0.05, 1000, 1000)

    assert (x1 - x0) == round(sf.CROP_MARGIN * 100)      # 1.20 * max(100, 50)


def test_crop_shifts_before_it_shrinks():
    """A box against the edge keeps its requested size, moved rather than clipped.

    Clipping the corners would silently return a smaller, non-square region; the
    protocol requires preserving the size where the image allows it.
    """

    interior = sf.square_crop(0.5, 0.5, 0.10, 0.10, 1000, 1000)
    at_edge = sf.square_crop(0.001, 0.5, 0.10, 0.10, 1000, 1000)

    assert (at_edge[2] - at_edge[0]) == (interior[2] - interior[0])
    assert at_edge[0] == 0


def test_crop_shrinks_only_when_the_image_cannot_supply_the_square():
    """Requested 1.20 * 2000 px in a 640x480 image -> the largest valid square."""

    x0, y0, x1, y1 = sf.square_crop(0.5, 0.5, 2.0, 2.0, 640, 480)

    assert (x1 - x0) == 480 == min(640, 480)
    assert y0 == 0 and y1 == 480


def test_crop_rejects_a_degenerate_image():
    with pytest.raises(ExportError, match="dimensions must be positive"):
        sf.square_crop(0.5, 0.5, 0.1, 0.1, 0, 480)


def test_crop_is_deterministic():
    first = sf.square_crop(0.37, 0.62, 0.083, 0.191, 613, 447)
    second = sf.square_crop(0.37, 0.62, 0.083, 0.191, 613, 447)

    assert first == second


# ------------------------------------------------------- pool-only filtering ---


def _fake_pool(path, *, n_images=4, per_image=3, include_eval=True):
    """A miniature pool file with both splits, so 'pool only' can be asserted."""

    images = [f"{i:012d}" for i in range(n_images)]
    pool_ids = np.repeat(images, per_image)
    pool_q = np.tile(np.arange(per_image), n_images)
    rows = pool_ids.size
    ids, queries, splits = list(pool_ids), list(pool_q), ["pool"] * rows
    if include_eval:
        eval_images = [f"{9000 + i:012d}" for i in range(2)]
        ids += list(np.repeat(eval_images, per_image))
        queries += list(np.tile(np.arange(per_image), 2))
        splits += ["eval"] * (2 * per_image)
    total = len(ids)
    np.savez(
        path,
        image_ids=np.asarray(ids, dtype=str),
        query_index=np.asarray(queries, dtype=np.int16),
        split=np.asarray(splits, dtype=str),
        boxes=np.tile(np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), (total, 1)),
    )
    return rows


def test_pool_rows_reads_only_the_pool_split(tmp_path, monkeypatch):
    """`eval` must never be loaded: fitting on it would leak the held-out split."""

    path = tmp_path / "mini.npz"
    expected = _fake_pool(path, n_images=4, per_image=3)
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", expected)
    monkeypatch.setattr(sf, "EXPECTED_IMAGES", 4)

    rows = sf.pool_rows(path)

    assert len(rows) == expected
    assert rows.images == [f"{i:012d}" for i in range(4)]
    assert not any(name.startswith("000000009") for name in rows.image_ids)
    # row_index points back into the *unfiltered* file
    assert rows.row_index.tolist() == list(range(expected))


def test_pool_rows_refuses_the_wrong_population(tmp_path):
    """80,000 / 1,600 is the frozen population; a different count is a failure."""

    path = tmp_path / "mini.npz"
    _fake_pool(path, n_images=4, per_image=3)

    with pytest.raises(ExportError, match="expected 80000"):
        sf.pool_rows(path)


def test_pool_rows_refuses_a_file_without_a_pool_split(tmp_path):
    path = tmp_path / "eval_only.npz"
    np.savez(
        path,
        image_ids=np.asarray(["a", "a"], dtype=str),
        query_index=np.asarray([0, 1], dtype=np.int16),
        split=np.asarray(["eval", "eval"], dtype=str),
        boxes=np.zeros((2, 4), dtype=np.float32),
    )

    with pytest.raises(ExportError, match="no rows with split"):
        sf.pool_rows(path)


def test_pool_rows_refuses_duplicate_proposal_identities(tmp_path, monkeypatch):
    path = tmp_path / "dupes.npz"
    np.savez(
        path,
        image_ids=np.asarray(["a", "a"], dtype=str),
        query_index=np.asarray([0, 0], dtype=np.int16),   # same identity twice
        split=np.asarray(["pool", "pool"], dtype=str),
        boxes=np.zeros((2, 4), dtype=np.float32),
    )
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", 2)
    monkeypatch.setattr(sf, "EXPECTED_IMAGES", 1)

    with pytest.raises(ExportError, match="duplicate"):
        sf.pool_rows(path)


def test_the_real_pool_is_the_frozen_population():
    """The committed file must be exactly 80,000 proposals over 1,600 images."""

    rows = sf.pool_rows(sf.POOL)

    assert len(rows) == 80_000
    assert len(rows.images) == 1_600
    assert np.unique(rows.keys).size == 80_000


# ---------------------------------------------------------- export round trip ---


def _export(rows: sf.PoolRows, *, count: int, dim=sf.FEATURE_DIM,
            normalise=True, seed=0) -> sf.SemanticExport:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(count, dim)).astype(np.float32)
    if normalise:
        features /= np.linalg.norm(features, axis=1, keepdims=True)
    return sf.SemanticExport(
        embeddings=features.astype(np.float16),
        keys=rows.keys[:count],
        image_ids=rows.image_ids[:count],
        query_index=rows.query_index[:count],
        row_index=rows.row_index[:count],
        provenance={"model_id": sf.MODEL_ID},
    )


@pytest.fixture
def mini(tmp_path, monkeypatch) -> sf.PoolRows:
    path = tmp_path / "mini.npz"
    expected = _fake_pool(path, n_images=4, per_image=3)
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", expected)
    monkeypatch.setattr(sf, "EXPECTED_IMAGES", 4)
    return sf.pool_rows(path)


def test_write_then_read_preserves_identity_and_provenance(tmp_path, mini):
    export = _export(mini, count=len(mini))
    path = sf.write(tmp_path / "features.npz", export)

    back = sf.read(path)

    assert np.array_equal(back.keys, mini.keys)
    assert np.array_equal(back.row_index, mini.row_index)
    assert back.provenance["model_id"] == sf.MODEL_ID
    assert back.features().dtype == np.float32       # stored small, evaluated in full


def test_read_refuses_a_different_export_version(tmp_path, mini):
    path = tmp_path / "old.npz"
    np.savez(
        path, embeddings=np.zeros((1, sf.FEATURE_DIM), np.float16),
        keys=mini.keys[:1], image_ids=mini.image_ids[:1],
        query_index=mini.query_index[:1], row_index=mini.row_index[:1],
        provenance=np.asarray("{}"),
        export_version=np.asarray("dinov2_vitb14_method_v2_v0"),
    )

    with pytest.raises(ExportError, match="not reinterpreted"):
        sf.read(path)


def test_export_refuses_mismatched_row_counts(mini):
    with pytest.raises(ExportError, match="rows against"):
        sf.SemanticExport(
            embeddings=np.zeros((2, sf.FEATURE_DIM), np.float16),
            keys=mini.keys[:3], image_ids=mini.image_ids[:3],
            query_index=mini.query_index[:3], row_index=mini.row_index[:3],
            provenance={},
        )


# ------------------------------------------------------------------- gates ---


def test_validate_accepts_a_correct_full_export(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))

    report = sf.validate(_export(mini, count=len(mini)), mini)

    assert report["rows"] == len(mini)
    assert report["dimension"] == sf.FEATURE_DIM
    assert report["worst_norm_deviation"] < sf.NORM_TOLERANCE


def test_validate_refuses_unnormalised_features(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))

    with pytest.raises(ExportError, match="not L2-normalised"):
        sf.validate(_export(mini, count=len(mini), normalise=False), mini)


def test_validate_refuses_zero_norm_rows(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))
    export = _export(mini, count=len(mini))
    embeddings = export.embeddings.copy()
    embeddings[1] = 0.0
    broken = sf.SemanticExport(
        embeddings=embeddings, keys=export.keys, image_ids=export.image_ids,
        query_index=export.query_index, row_index=export.row_index, provenance={},
    )

    with pytest.raises(ExportError, match="zero-norm"):
        sf.validate(broken, mini)


def test_validate_refuses_non_finite_features(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))
    export = _export(mini, count=len(mini))
    embeddings = export.embeddings.copy()
    embeddings[0, 0] = np.float16("nan")
    broken = sf.SemanticExport(
        embeddings=embeddings, keys=export.keys, image_ids=export.image_ids,
        query_index=export.query_index, row_index=export.row_index, provenance={},
    )

    with pytest.raises(ExportError, match="non-finite"):
        sf.validate(broken, mini)


def test_validate_refuses_a_wrong_feature_dimension(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))

    with pytest.raises(ExportError, match="expected 768"):
        sf.validate(_export(mini, count=len(mini), dim=384), mini)


def test_validate_refuses_rows_out_of_pool_order(mini, monkeypatch):
    """An audit joining by position would compare different proposals silently."""

    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))
    export = _export(mini, count=len(mini))
    shuffled = sf.SemanticExport(
        embeddings=export.embeddings, keys=export.keys[::-1],
        image_ids=export.image_ids[::-1], query_index=export.query_index[::-1],
        row_index=export.row_index[::-1], provenance={},
    )

    with pytest.raises(ExportError, match="not in the pool's own order"):
        sf.validate(shuffled, mini)


def test_validate_refuses_an_incomplete_full_export(mini, monkeypatch):
    monkeypatch.setattr(sf, "EXPECTED_PROPOSALS", len(mini))

    with pytest.raises(ExportError, match="rows against"):
        sf.validate(_export(mini, count=len(mini) - 1), mini)


def test_smoke_subset_passes_the_correctness_gates_without_the_size_gate(mini):
    """`full=False` relaxes population size only; every other check still bites."""

    subset = _export(mini, count=3)

    report = sf.validate(subset, mini, full=False)
    assert report["rows"] == 3

    unnormalised = _export(mini, count=3, normalise=False)
    with pytest.raises(ExportError, match="not L2-normalised"):
        sf.validate(unnormalised, mini, full=False)


def test_crop_specification_is_recorded_as_data(mini):
    """Provenance must carry the crop, not a prose description of it."""

    specification = sf.crop_specification()

    assert specification["margin"] == sf.CROP_MARGIN
    assert specification["padding"] == "none -- real image pixels only"
    assert specification["resize"].startswith(f"{sf.CROP_SIZE}x{sf.CROP_SIZE}")


def test_keys_are_shared_with_the_decoder_layer_export(mini):
    """One identity convention across both exports, so rows cannot disagree."""

    assert np.array_equal(
        mini.keys, proposal_keys(mini.image_ids, mini.query_index)
    )


# ---------------------------------------------------- deterministic P0/P1/P2 ---


def test_populations_are_deterministic_and_match_the_frozen_references():
    """P1/P2 must be reproducible and reuse the repository's own definitions.

    Reused rather than redefined: ``admissible_mask`` sorts stably and ``nms_keep``
    walks a stable order, so the same pool yields the same masks. The reference
    counts come from the committed decoder-layer audit; a silent redefinition of
    P1 or P2 would make the two audits incomparable while still producing
    plausible numbers.
    """

    import sys

    import numpy as np

    from owl import proposals as proposals_module
    from tools.audit_decoder_layers import populations
    from tools.diagnose_representation import load

    sys.path.insert(0, str(sf.POOL.parent.parent.parent))
    pool = load()
    payload = np.load(sf.POOL, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == sf.POOL_SPLIT
    pool["raw_boxes"] = payload["boxes"][keep].astype(np.float32)
    candidates = proposals_module.from_frozen_pool(sf.POOL, split=sf.POOL_SPLIT)

    first = populations(pool, candidates)
    second = populations(pool, candidates)

    assert list(first) == ["P0_raw", "P1_admissible", "P2_admissible_nms"]
    for name in first:
        assert np.array_equal(first[name], second[name]), f"{name} is not deterministic"

    assert int(first["P0_raw"].sum()) == 80_000
    assert int(first["P1_admissible"].sum()) == 24_000
    assert int(first["P2_admissible_nms"].sum()) == 15_518

    background = pool["kind"] == "background"
    shares = {name: float(background[mask].mean()) for name, mask in first.items()}
    assert shares["P0_raw"] == pytest.approx(0.814, abs=0.001)
    assert shares["P1_admissible"] == pytest.approx(0.652, abs=0.001)
    assert shares["P2_admissible_nms"] == pytest.approx(0.767, abs=0.001)
