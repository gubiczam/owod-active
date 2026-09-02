"""The decoder-layer export contract: keys, alignment, and the validation gate.

The gate is the point. An export whose ``hs[5]`` does not reproduce the committed
pool is describing an unknown tensor -- a wrong checkpoint, a wrong reconstruction
of PROB's arguments, a hook on the wrong module, a different image order, or a
mis-joined key -- and every per-layer number computed from it would be
meaningless while looking perfectly plausible. So it raises rather than warns, and
these tests pin each way it can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from owl import decoder_layers as dl


def _export(features: np.ndarray, keys: np.ndarray,
            layers: tuple[int, ...] = tuple(range(6))) -> dl.LayerExport:
    return dl.LayerExport(features=features.astype(np.float16), keys=keys,
                          layer_indices=layers, provenance={})


@pytest.fixture
def pool_like() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three images x four queries, six layers, deterministic."""

    generator = np.random.default_rng(0)
    images = np.repeat(np.array(["a", "b", "c"]), 4)
    queries = np.tile(np.arange(4), 3)
    keys = dl.proposal_keys(images, queries)
    features = generator.normal(size=(6, keys.size, 8)).astype(np.float32)
    return features, keys, features[dl.FINAL_LAYER]


# ------------------------------------------------------------------- keys ---


def test_keys_identify_a_proposal_uniquely():
    keys = dl.proposal_keys(np.array(["im1", "im1", "im2"]), np.array([0, 7, 0]))

    assert keys.tolist() == ["im1#0", "im1#7", "im2#0"]
    assert np.unique(keys).size == 3


def test_keys_do_not_collide_across_images_and_queries():
    """'a#11' must not be reachable from ('a1', 1) -- the separator has to bite."""

    left = dl.proposal_keys(np.array(["a"]), np.array([11]))
    right = dl.proposal_keys(np.array(["a1"]), np.array([1]))
    assert left[0] != right[0]


# -------------------------------------------------------------- alignment ---


def test_align_reorders_to_the_target_order(pool_like):
    _, keys, _ = pool_like
    shuffled = keys[::-1]

    rows = dl.align(shuffled, keys)

    assert np.array_equal(shuffled[rows], keys)


def test_align_refuses_a_partial_export(pool_like):
    """Auditing a subset would compare layers on different candidates."""

    _, keys, _ = pool_like

    with pytest.raises(dl.ExportError, match="absent from the export"):
        dl.align(keys[:-1], keys)


# ------------------------------------------------------------------- gate ---


def test_validate_passes_when_the_final_layer_matches(pool_like, tmp_path):
    features, keys, reference = pool_like

    similarity = dl.validate(_export(features, keys), reference)

    assert similarity > dl.VALIDATION_SIMILARITY


def test_validate_refuses_a_wrong_layer(pool_like):
    """Exporting hs[4] in hs[5]'s slot must be caught, not averaged over."""

    features, keys, reference = pool_like
    swapped = features.copy()
    swapped[dl.FINAL_LAYER] = features[4]

    with pytest.raises(dl.ExportError, match="Refusing to audit"):
        dl.validate(_export(swapped, keys), reference)


def test_validate_refuses_a_shuffled_join(pool_like):
    """A mis-joined key is the failure most likely to look plausible."""

    features, keys, reference = pool_like
    shuffled = features.copy()
    shuffled[dl.FINAL_LAYER] = features[dl.FINAL_LAYER][::-1]

    with pytest.raises(dl.ExportError, match="Refusing to audit"):
        dl.validate(_export(shuffled, keys), reference)


def test_validate_refuses_an_export_without_the_final_layer(pool_like):
    features, keys, reference = pool_like

    with pytest.raises(dl.ExportError, match="omits layer"):
        dl.validate(_export(features[:5], keys, layers=(0, 1, 2, 3, 4)), reference)


def test_validate_reports_a_shape_mismatch_rather_than_broadcasting(pool_like):
    features, keys, reference = pool_like

    with pytest.raises(dl.ExportError, match="the pool's embeddings are"):
        dl.validate(_export(features, keys), reference[:, :4])


def test_float16_round_trip_still_passes_the_gate(pool_like):
    """The threshold must tolerate the storage format it mandates."""

    features, keys, reference = pool_like
    stored = features.astype(np.float16).astype(np.float32)

    assert dl.validate(_export(stored, keys), reference) > dl.VALIDATION_SIMILARITY


# ------------------------------------------------------------- round trip ---


def test_write_then_read_preserves_layers_and_provenance(pool_like, tmp_path):
    features, keys, reference = pool_like
    path = dl.write(tmp_path / "layers.npz", features, keys, tuple(range(6)),
                    {"checkpoint_sha256": "abc", "gpu": "T4"})

    export = dl.read(path)

    assert export.layer_indices == tuple(range(6))
    assert export.provenance["gpu"] == "T4"
    assert np.array_equal(export.keys, keys)
    assert dl.validate(export, reference) > dl.VALIDATION_SIMILARITY


def test_read_refuses_a_different_export_version(pool_like, tmp_path):
    """An older file must not be reinterpreted under new semantics."""

    features, keys, _ = pool_like
    path = tmp_path / "old.npz"
    np.savez(path, features=features.astype(np.float16), keys=keys,
             layer_indices=np.arange(6), provenance=np.asarray("{}"),
             export_version=np.asarray("decoder_layers_v0"))

    with pytest.raises(dl.ExportError, match="not reinterpreted"):
        dl.read(path)


def test_layer_lookup_rejects_a_layer_not_exported(pool_like):
    features, keys, _ = pool_like
    export = _export(features[:3], keys, layers=(0, 1, 2))

    with pytest.raises(dl.ExportError, match="not in this export"):
        export.layer(5)


def test_shape_disagreement_is_caught_at_construction(pool_like):
    features, keys, _ = pool_like

    with pytest.raises(dl.ExportError, match="declared layer indices"):
        _export(features, keys, layers=(0, 1, 2))
    with pytest.raises(dl.ExportError, match="rows against"):
        _export(features[:, :-1], keys)


# ------------------------------------------------- the cwd guard (PROB build) ---


def test_working_directory_restores_cwd_on_success(tmp_path):
    """PROB's backbone loads 'models/dino_resnet50_pretrain.pth' relatively, so
    construction must run inside the checkout -- and must not leave it there."""

    import os

    from tools.export_decoder_layers import working_directory

    before = os.getcwd()
    with working_directory(tmp_path):
        assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path)
    assert os.getcwd() == before


def test_working_directory_restores_cwd_when_the_block_raises(tmp_path):
    """The failure that matters: a build that dies must not strand the process.

    Without the ``finally`` the next notebook cell fails somewhere unrelated,
    which is far harder to read than the original error.
    """

    import os

    from tools.export_decoder_layers import working_directory

    before = os.getcwd()
    with pytest.raises(FileNotFoundError, match="dino"):
        with working_directory(tmp_path):
            raise FileNotFoundError("models/dino_resnet50_pretrain.pth")
    assert os.getcwd() == before


def test_preflight_names_the_relative_path_initialisation(tmp_path):
    """A missing backbone init must be reported before a GPU session is spent."""

    from owl.decoder_layers import ExportError
    from tools.export_decoder_layers import preflight

    prob = tmp_path / "PROB"
    (prob / "models").mkdir(parents=True)
    (prob / "main_open_world.py").touch()
    (prob / "models" / "__init__.py").touch()
    data = tmp_path / "data"
    (data / "JPEGImages").mkdir(parents=True)
    (data / "Annotations").mkdir(parents=True)
    checkpoint = tmp_path / "t1.pth"
    checkpoint.touch()

    with pytest.raises(ExportError, match="dino_resnet50_pretrain.pth"):
        preflight(prob, checkpoint, data)

    (prob / "models" / "dino_resnet50_pretrain.pth").touch()
    preflight(prob, checkpoint, data)          # now passes


def test_preflight_reports_every_problem_at_once(tmp_path):
    """One round trip per session, not one per missing file."""

    from owl.decoder_layers import ExportError
    from tools.export_decoder_layers import preflight

    with pytest.raises(ExportError) as error:
        preflight(tmp_path / "nope", tmp_path / "missing.pth", tmp_path / "nodata")
    message = str(error.value)
    assert "main_open_world.py" in message
    assert "checkpoint" in message
    assert "JPEGImages" in message
