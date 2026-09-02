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
import contextlib
import importlib
import json
import os
import platform
import re
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


#: Top-level package names PROB owns that also exist on PyPI. ``datasets`` is the
#: dangerous one: Hugging Face's ``datasets`` is commonly pre-installed in Colab,
#: and if it is already in ``sys.modules`` then putting PROB first on ``sys.path``
#: does nothing -- the cached module wins, and the failure is
#: ``ModuleNotFoundError: No module named 'datasets.<x>'``, which looks exactly
#: like a missing file in PROB rather than a shadowed package.
PROB_TOP_LEVEL = ("datasets", "models", "util", "engine")


def _under(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def purge_shadowing_modules(prob_root: Path) -> list[str]:
    """Evict cached top-level modules that would shadow PROB's own packages.

    ``sys.path`` order is irrelevant once a name is in ``sys.modules``, so a
    pre-imported Hugging Face ``datasets`` cannot be displaced by inserting
    ``prob_root`` at the front. Anything under ``prob_root`` is left alone.
    """

    evicted = []
    for name in sorted(sys.modules):
        if name.split(".")[0] not in PROB_TOP_LEVEL:
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        search = list(getattr(module, "__path__", []) or [])
        if _under(origin, prob_root) or any(_under(entry, prob_root) for entry in search):
            continue
        del sys.modules[name]
        evicted.append(f"{name} (was {origin or search or 'namespace package'})")
    return evicted


def prob_import(prob_root: Path, dotted: str):
    """Import ``dotted`` from the PROB checkout, or refuse.

    Puts ``prob_root`` first on ``sys.path`` -- not merely present, first, since a
    stale earlier entry would keep winning -- and then verifies that the module
    which actually resolved lives inside ``prob_root``. Without that check an
    installed package of the same name can satisfy the import silently, and every
    number downstream would come from code nobody pinned.
    """

    root = str(Path(prob_root).resolve())
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    module = importlib.import_module(dotted)
    origin = getattr(module, "__file__", None)
    if not _under(origin, prob_root):
        raise dl.ExportError(
            f"'{dotted}' resolved to {origin!r}, which is not under {prob_root}. "
            "An installed package of the same name satisfied the import. Refusing "
            "to run PROB code that is not the pinned checkout."
        )
    return module, origin


def find_module_defining(prob_root: Path, symbol: str,
                         packages: tuple[str, ...] = ("datasets", "models")) -> list[str]:
    """Dotted paths of modules in the checkout that define ``symbol``.

    The import path is read off the actual source tree rather than assumed. This
    exists because ``datasets/open_world.py`` does not exist in this checkout --
    ``make_coco_transforms`` is defined elsewhere -- and guessing produced a
    ``ModuleNotFoundError`` indistinguishable from a shadowed package.
    """

    pattern = re.compile(rf"^def\s+{re.escape(symbol)}\s*\(", re.MULTILINE)
    found = []
    for package in packages:
        directory = Path(prob_root) / package
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                relative = path.relative_to(prob_root).with_suffix("")
                found.append(".".join(relative.parts))
    return found


@contextlib.contextmanager
def working_directory(path: Path):
    """Run a block with ``path`` as the process cwd, restoring it whatever happens.

    PROB's backbone loads its self-supervised initialisation through a **relative**
    path -- ``torch.load('models/dino_resnet50_pretrain.pth')`` -- so the model can
    only be constructed from inside the PROB checkout. Every other path this tool
    handles is absolute, so the change is scoped to construction alone rather than
    to the whole run: a global chdir would silently reinterpret ``--out`` and
    ``--pool`` if either were ever passed as a relative path.

    ``finally`` rather than a plain pair of calls because a failed build must not
    leave the interpreter in another directory -- the next cell in a notebook
    would then fail somewhere unrelated, which is the hard kind of bug to read.
    """

    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def preflight(prob_root: Path, checkpoint: Path, data_root: Path) -> None:
    """Assert what must exist before a GPU session is spent on discovering it."""

    problems = []
    if not (prob_root / "main_open_world.py").is_file():
        problems.append(f"{prob_root}/main_open_world.py missing -- not a PROB checkout")
    if not (prob_root / "models" / "__init__.py").is_file():
        problems.append(f"{prob_root}/models/__init__.py missing")
    # the relative-path initialisation the cwd fix exists for
    backbone = prob_root / "models" / "dino_resnet50_pretrain.pth"
    if not backbone.is_file():
        problems.append(
            f"{backbone} missing. PROB loads it by the relative path "
            "'models/dino_resnet50_pretrain.pth' during backbone construction, so "
            "it must sit inside the PROB checkout."
        )
    if not Path(checkpoint).is_file():
        problems.append(f"checkpoint {checkpoint} missing")
    if not (Path(data_root) / "JPEGImages").is_dir():
        problems.append(f"{data_root}/JPEGImages missing -- run materialize_pool_images.py")
    if not (Path(data_root) / "Annotations").is_dir():
        problems.append(f"{data_root}/Annotations missing -- run materialize_pool_images.py")
    if problems:
        raise dl.ExportError("preflight failed:\n  " + "\n  ".join(problems))
    print("[preflight] PROB checkout, backbone init, checkpoint and data root all present")

    # Read the import path off the tree instead of assuming it. This checkout has
    # no datasets/open_world.py -- make_coco_transforms lives elsewhere -- and the
    # ModuleNotFoundError that produced was indistinguishable from a shadowed
    # Hugging Face `datasets`.
    where = find_module_defining(prob_root, "make_coco_transforms")
    if not where:
        raise dl.ExportError(
            f"no module under {prob_root}/datasets or /models defines "
            "make_coco_transforms(). The checkout layout is not the expected one."
        )
    print(f"[preflight] make_coco_transforms() defined in: {', '.join(where)}")

    evicted = purge_shadowing_modules(prob_root)
    if evicted:
        print("[preflight] evicted shadowing modules so PROB's own are importable:")
        for entry in evicted:
            print(f"             {entry}")
    else:
        print("[preflight] no shadowing top-level modules were cached")


def build_model(prob_root: Path, checkpoint: Path, device: str):
    """Build PROB and load the checkpoint strictly. Returns (model, chosen args)."""

    import torch

    purge_shadowing_modules(prob_root)
    parser_module, parser_origin = prob_import(prob_root, "main_open_world")
    models_module, models_origin = prob_import(prob_root, "models")
    print(f"[import] main_open_world      <- {parser_origin}")
    print(f"[import] models               <- {models_origin}")
    get_args_parser = parser_module.get_args_parser
    prob_build_model = models_module.build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("model", state)

    entry_cwd = os.getcwd()
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
            # cwd must be the PROB checkout: the backbone's DINO initialisation is
            # loaded by a relative path during construction.
            with working_directory(prob_root):
                model = prob_build_model(args, mode=args.model_type)[0]
                model.load_state_dict(weights, strict=True)
        except (RuntimeError, KeyError, TypeError, FileNotFoundError) as error:
            failures.append(f"{candidate}: {str(error)[:300]}")
            continue
        print(f"[build] strict load succeeded with {candidate}")
        if os.getcwd() != entry_cwd:
            raise dl.ExportError(
                f"cwd is {os.getcwd()} after construction, expected {entry_cwd}; "
                "the working_directory guard did not restore it."
            )
        return model.to(device).eval(), args

    if os.getcwd() != entry_cwd:
        os.chdir(entry_cwd)
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
    parser.add_argument("--smoke-images", type=int, default=0,
                        help="run the full path on N images, apply the gate to their "
                             "own pool rows, write nothing, and exit")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    preflight(Path(arguments.prob_root), Path(arguments.checkpoint),
              Path(arguments.data_root))

    out = Path(arguments.out)
    if out.exists() and not arguments.smoke_images:
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
    #
    # Module paths are resolved against the checkout and each resolved file is
    # verified to live inside it. `datasets` is the name that matters: Hugging
    # Face ships a package of that name, it is often already imported under
    # Colab, and a cached one cannot be displaced by sys.path order alone.
    prob_root = Path(arguments.prob_root)
    dataset_module, dataset_origin = prob_import(
        prob_root, "datasets.torchvision_datasets.open_world")
    OWDetection = dataset_module.OWDetection
    print(f"[import] OWDetection          <- {dataset_origin}")

    make_coco_transforms = None
    candidates = find_module_defining(prob_root, "make_coco_transforms")
    for dotted in candidates:
        module, origin = prob_import(prob_root, dotted)
        function = getattr(module, "make_coco_transforms", None)
        if function is not None:
            make_coco_transforms = function
            print(f"[import] make_coco_transforms <- {origin}")
            break
    if make_coco_transforms is None:
        raise dl.ExportError(
            f"none of {candidates} exposed make_coco_transforms after import"
        )

    # The split *name* decides which annotation filters PROB runs, by substring.
    # owl.evaluation_subset.check_split_name refuses anything but a test-only
    # marker, so the transform applied here is the inference one. The target is
    # discarded -- annotations come from the committed pool -- but the image
    # preprocessing must be PROB's own or hs[5] will not reproduce the pool, which
    # is exactly what the gate below would catch.
    from owl.evaluation_subset import check_split_name
    check_split_name(arguments.image_set)

    # OWDetection subscripts `transforms`: __getitem__ reads `self.transforms[0]`
    # as the image-set *string* to pick the annotation filter, and calls
    # `self.transforms[-1](img, target)`. So it wants a (marker, callable) pair,
    # not a bare transform -- and the marker being a substring match is the same
    # trap owl.evaluation_subset guards. The bare form is tried as a fallback
    # only because a fork may differ; either way the smoke test below decides.
    dataset = None
    failures = []
    for description, transforms in (
        ("(marker, callable) pair", (arguments.image_set, make_coco_transforms("test"))),
        ("bare callable", make_coco_transforms("test")),
    ):
        try:
            candidate = OWDetection(
                args, Path(arguments.data_root), image_set=arguments.image_set,
                dataset="OWDETR", transforms=transforms,
            )
            # Pull one sample now. A convention mismatch must cost seconds, not
            # 1,600 images: this is the only place it can be detected cheaply.
            sample = candidate[0][0]
            if not isinstance(sample, torch.Tensor) or sample.ndim != 3:
                raise TypeError(
                    f"sample 0 came back as {type(sample).__name__} "
                    f"{getattr(sample, 'shape', '')}, not a 3-d image tensor"
                )
            dataset = candidate
            print(f"[dataset] {description}: sample 0 is {tuple(sample.shape)}")
            break
        except Exception as error:            # noqa: BLE001 - report and try the next form
            failures.append(f"{description}: {type(error).__name__}: {str(error)[:200]}")
    if dataset is None:
        raise dl.ExportError(
            "PROB's OWDetection would not yield an image tensor under either "
            "transforms convention. Refusing to export.\n  " + "\n  ".join(failures)
        )

    # `imgids` holds integers (convert_image_id(..., to_integer=True)); the raw
    # zero-padded string ids the pool uses live in `image_set`.
    names = [str(name) for name in dataset.image_set]
    index_of = {name: position for position, name in enumerate(names)}
    if len(index_of) != len(names):
        raise dl.ExportError(
            f"the dataset lists {len(names)} images but only {len(index_of)} are "
            "distinct; a duplicated id would make the key join ambiguous."
        )
    missing = [name for name in image_list if name not in index_of]
    if missing:
        raise dl.ExportError(
            f"{len(missing)} pool images are absent from the dataset index "
            f"(first: {missing[:3]}). Check that "
            f"ImageSets/OWDETR/{arguments.image_set}.txt was written by "
            "tools/materialize_pool_images.py."
        )

    started = time.time()
    misc_module, misc_origin = prob_import(prob_root, "util.misc")
    nested_tensor_from_tensor_list = misc_module.nested_tensor_from_tensor_list
    print(f"[import] util.misc            <- {misc_origin}")

    def run_images(names: list[str], *, label: str) -> dict[str, np.ndarray]:
        """One forward pass per batch; returns {image_id: (layers, queries, dim)}."""

        out: dict[str, np.ndarray] = {}
        begun = time.time()
        with torch.no_grad():
            for position in range(0, len(names), arguments.batch_size):
                batch = names[position:position + arguments.batch_size]
                tensors = [dataset[index_of[name]][0].to(arguments.device)
                           for name in batch]
                model(nested_tensor_from_tensor_list(tensors))
                # store[l] is (batch, queries, dim); one row per (image, query)
                block = torch.stack(
                    [store[l] for l in range(dl.N_DECODER_LAYERS)]
                ).float().cpu().numpy()
                for offset, name in enumerate(batch):
                    out[name] = block[:, offset].astype(np.float16)
                if position % (arguments.batch_size * 50) == 0:
                    done = position + len(batch)
                    rate = done / max(time.time() - begun, 1e-6)
                    print(f"[{label}] {done:,}/{len(names):,} images "
                          f"({rate:.1f} img/s, eta "
                          f"{(len(names) - done) / max(rate, 1e-6) / 60:.1f} min)")
        return out

    # ---- smoke test: the full path on a handful of images, gated the same way --
    #
    # A shape check alone would pass while the join was wrong, so this runs the
    # real gate on the subset: hs[5] for these images' own pool rows must
    # reproduce the pool's committed embeddings. Everything the full run can get
    # wrong -- checkpoint, reconstructed arguments, cwd, hooks, transforms
    # convention, key join -- is exercised here, in seconds instead of 25 minutes.
    if arguments.smoke_images:
        names = image_list[:arguments.smoke_images]
        produced = run_images(names, label="smoke")
        for handle in handles:
            handle.remove()
        block = produced[names[0]]
        print(f"[smoke] per-image block {block.shape} = (layers, queries, dim)")
        if block.shape[0] != dl.N_DECODER_LAYERS:
            raise dl.ExportError(
                f"captured {block.shape[0]} layers, expected {dl.N_DECODER_LAYERS}"
            )
        n_queries = block.shape[1]
        subset = np.isin(pool_images, names)
        keys = dl.proposal_keys(
            np.repeat(np.asarray(names, dtype=str), n_queries),
            np.tile(np.arange(n_queries), len(names)),
        )
        stacked = np.concatenate(
            [produced[name].transpose(1, 0, 2) for name in names], axis=0
        )
        rows = dl.align(keys, pool_keys[subset])
        pool = proposals_module.from_frozen_pool(arguments.pool, split="pool")
        similarity = dl.validate(
            dl.LayerExport(
                features=stacked[rows].transpose(1, 0, 2),
                keys=pool_keys[subset],
                layer_indices=tuple(range(dl.N_DECODER_LAYERS)),
                provenance={},
            ),
            pool.embeddings[subset],
        )
        print(f"[smoke] {int(subset.sum())} pool proposals over {len(names)} image(s); "
              f"hs[5] reproduces the pool at mean cosine {similarity:.6f}  PASS")
        print("[smoke] the full export is safe to run")
        return

    collected = run_images(image_list, label="export")
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
