#!/usr/bin/env python
"""Export hs[0..5] for the committed pool's proposals. GPU, inference only.

Runs in Colab. It does **not** modify PROB and does not touch the existing
bridge: the six decoder-layer hidden states are captured with forward hooks on
``transformer.decoder.layers[0..5]``, whose return value the architecture trace in
``docs/decoder_layer_protocol_2026-09-02.md`` establishes is exactly the tensor
appended to ``intermediate`` and returned as ``hs[l]`` (post the layer's own
``norm3``; no further normalisation is applied).

Hooks rather than a patch for three reasons: this session cannot read the fork's
bridge, a patch would have to be kept in sync with it, and a hook cannot change
the numbers the model produces.

**Model arguments are reconstructed from PROB's own argparse and then checked
against the checkpoint, not assumed.** ``--with_box_refine`` and ``--two_stage``
are absent from the published ``S_OWOD_BENCHMARK.sh`` t1 stage, so they arrive
through ``PY_ARGS`` and cannot be read off the config. Rather than guess, this
tries the plausible combinations and keeps the one whose ``state_dict`` loads
**strictly**. A wrong guess therefore fails loudly instead of silently loading a
differently-shaped model.

The export is gated: ``hs[5]`` must reproduce the pool's committed ``embeddings``.

    python tools/export_decoder_layers.py \
        --prob-root /content/PROB --data-root /content/data \
        --checkpoint /content/drive/MyDrive/OWL/checkpoints/SOWODB/t1.pth \
        --out /content/drive/MyDrive/OWL/features/decoder_layers_v1.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl import decoder_layers as dl
from owl import proposals as proposals_module

POOL = Path(__file__).resolve().parent.parent / "data" / "pool" / "sowodb_t1_frozen_pool.npz"

#: Reconstruction candidates, most likely first. The published t1 stage passes
#: neither flag explicitly, so both settings are plausible; the checkpoint decides.
ARCHITECTURE_CANDIDATES = (
    {"with_box_refine": True, "two_stage": False},
    {"with_box_refine": False, "two_stage": False},
    {"with_box_refine": True, "two_stage": True},
    {"with_box_refine": False, "two_stage": True},
)


def build_model(prob_root: Path, checkpoint: Path, device: str):
    """Build PROB and load the checkpoint strictly. Returns (model, chosen args)."""

    sys.path.insert(0, str(prob_root))
    import torch
    from main_open_world import get_args_parser
    from models import build_model as prob_build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("model", state)

    failures = []
    for candidate in ARCHITECTURE_CANDIDATES:
        args = get_args_parser().parse_args([])
        args.dataset = "OWDETR"
        args.model_type = "prob"
        args.PREV_INTRODUCED_CLS = 0
        args.CUR_INTRODUCED_CLS = 19
        args.obj_temp = 1.3
        args.device = device
        for name, value in candidate.items():
            setattr(args, name, value)
        try:
            model = prob_build_model(args, mode=args.model_type)[0]
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, KeyError, TypeError) as error:
            failures.append(f"{candidate}: {str(error)[:300]}")
            continue
        print(f"[build] strict load succeeded with {candidate}")
        return model.to(device).eval(), args

    raise dl.ExportError(
        "No reconstruction of PROB's arguments loaded this checkpoint strictly. "
        "Refusing to export from a model whose architecture is not the trained "
        "one.\n  " + "\n  ".join(failures)
    )


def attach_hooks(model, n_layers: int):
    """Forward hooks on each decoder layer. Returns (store, handles)."""

    core = getattr(model, "module", model)
    decoder = core.transformer.decoder
    layers = decoder.layers
    if len(layers) != n_layers:
        raise dl.ExportError(
            f"the decoder has {len(layers)} layers; the protocol declares {n_layers}"
        )

    store: dict[int, object] = {}

    def make(index: int):
        def hook(_module, _inputs, output):
            # a decoder layer returns the post-norm3 hidden state; some variants
            # return a tuple, so take the first tensor either way
            tensor = output[0] if isinstance(output, tuple) else output
            store[index] = tensor.detach()
        return hook

    return store, [layer.register_forward_hook(make(index))
                   for index, layer in enumerate(layers)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--image-set", default="owl_layer_test",
                        help="ImageSets/OWDETR/<name>.txt written by "
                             "tools/materialize_pool_images.py")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    out = Path(arguments.out)
    if out.exists():
        print(f"[resume] {out} exists; verifying instead of re-exporting")
        export = dl.read(out)
        pool = proposals_module.from_frozen_pool(arguments.pool, split="pool")
        print(f"[resume] validation similarity {dl.validate(export, pool.embeddings):.6f}")
        return

    import torch

    # ---- the pool defines exactly which proposals to export -----------------
    payload = np.load(arguments.pool, allow_pickle=True)
    keep = np.asarray(payload["split"], dtype=str) == "pool"
    pool_images = np.asarray(payload["image_ids"], dtype=str)[keep]
    pool_queries = np.asarray(payload["query_index"])[keep].astype(np.int64)
    pool_keys = dl.proposal_keys(pool_images, pool_queries)
    image_list = sorted(set(pool_images.tolist()))
    print(f"[pool] {pool_keys.size:,} proposals over {len(image_list):,} images")

    model, args = build_model(Path(arguments.prob_root), Path(arguments.checkpoint),
                              arguments.device)
    store, handles = attach_hooks(model, dl.N_DECODER_LAYERS)

    # ---- PROB's own dataset, so preprocessing matches training --------------
    sys.path.insert(0, str(Path(arguments.prob_root)))
    from datasets.open_world import make_coco_transforms
    from datasets.torchvision_datasets.open_world import OWDetection

    # The split *name* decides which annotation filters PROB runs, by substring.
    # owl.evaluation_subset.check_split_name refuses anything but a test-only
    # marker, so the transform applied here is the inference one. The target is
    # discarded -- annotations come from the committed pool -- but the image
    # preprocessing must be PROB's own or hs[5] will not reproduce the pool, which
    # is exactly what the gate below would catch.
    from owl.evaluation_subset import check_split_name
    check_split_name(arguments.image_set)
    dataset = OWDetection(
        args, Path(arguments.data_root), image_set=arguments.image_set,
        dataset="OWDETR", transforms=make_coco_transforms("test"),
    )
    index_of = {name: position for position, name in enumerate(dataset.imgids)}
    missing = [name for name in image_list if name not in index_of]
    if missing:
        raise dl.ExportError(
            f"{len(missing)} pool images are absent from the dataset index "
            f"(first: {missing[:3]}). The data root does not hold the pool's images."
        )

    collected: dict[str, np.ndarray] = {}
    started = time.time()
    with torch.no_grad():
        for position in range(0, len(image_list), arguments.batch_size):
            batch = image_list[position:position + arguments.batch_size]
            tensors = [dataset[index_of[name]][0].to(arguments.device) for name in batch]
            sizes = torch.stack([
                torch.tensor(t.shape[-2:], device=arguments.device) for t in tensors
            ])
            from util.misc import nested_tensor_from_tensor_list
            model(nested_tensor_from_tensor_list(tensors))

            # store[l] is (batch, queries, dim); one row per (image, query)
            block = torch.stack([store[l] for l in range(dl.N_DECODER_LAYERS)])
            block = block.float().cpu().numpy()
            for offset, name in enumerate(batch):
                collected[name] = block[:, offset].astype(np.float16)
            del sizes
            if position % (arguments.batch_size * 50) == 0:
                done = position + len(batch)
                rate = done / max(time.time() - started, 1e-6)
                print(f"[export] {done:,}/{len(image_list):,} images "
                      f"({rate:.1f} img/s, eta {(len(image_list) - done) / max(rate, 1e-6) / 60:.1f} min)")
    for handle in handles:
        handle.remove()

    # ---- assemble in the pool's own row order ------------------------------
    n_queries = next(iter(collected.values())).shape[1]
    export_keys = dl.proposal_keys(
        np.repeat(np.asarray(image_list, dtype=str), n_queries),
        np.tile(np.arange(n_queries), len(image_list)),
    )
    stacked = np.concatenate(
        [collected[name].transpose(1, 0, 2) for name in image_list], axis=0
    )  # (images*queries, layers, dim)
    rows = dl.align(export_keys, pool_keys)
    features = stacked[rows].transpose(1, 0, 2)  # (layers, N, dim)
    print(f"[export] features {features.shape} float16 "
          f"({features.nbytes / 1e6:.0f} MB)")

    provenance = {
        "export_version": dl.EXPORT_VERSION,
        "layer_indices": list(range(dl.N_DECODER_LAYERS)),
        "checkpoint_sha256": dl.sha256(arguments.checkpoint),
        "pool_sha256": dl.sha256(arguments.pool),
        "architecture": {"with_box_refine": args.with_box_refine,
                         "two_stage": args.two_stage,
                         "num_queries": args.num_queries,
                         "dec_layers": args.dec_layers,
                         "hidden_dim": args.hidden_dim},
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "images": len(image_list),
        "proposals": int(pool_keys.size),
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    dl.write(out, features, pool_keys, tuple(range(dl.N_DECODER_LAYERS)), provenance)
    export = dl.read(out)
    pool = proposals_module.from_frozen_pool(arguments.pool, split="pool")
    similarity = dl.validate(export, pool.embeddings)
    provenance["validation_similarity_hs5_vs_pool"] = similarity
    dl.write(out, features, pool_keys, tuple(range(dl.N_DECODER_LAYERS)), provenance)

    Path(str(out) + ".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"\n[gate] hs[5] reproduces the pool at mean cosine {similarity:.6f}  PASS")
    print(f"[done] {out}  in {provenance['wall_clock_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
