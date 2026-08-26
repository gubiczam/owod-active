"""Calling PROB. The only place in this package that knows about a GPU.

PROB is not vendored here. It is driven through ``daowod_prob_bridge.py``, a CLI
that lives on the ``feat/daowod-bridge-v2`` branch of ``gubiczam/PROB`` and
exposes three verbs:

``predict``   run the detector over a list of image ids and export, per
              proposal: box, 256-d decoder embedding, class posterior and
              objectness. This is what fills a :class:`~owl.proposals.Candidates`.
``train``     fine-tune from a previous checkpoint on a list of labelled image
              ids, optionally with ``--replay-ids``, and write a new checkpoint.
``evaluate``  score a checkpoint with PROB's own official evaluator and write
              the metrics JSON that :mod:`owl.metrics` reads.

Every call is **resumable**: if the output already exists the call is skipped and
reported as cached. A Colab session that dies halfway through a ten-task chain
resumes at the task it died on, which is the difference between a usable
protocol and one that needs an uninterrupted eight hours.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

PROB_REPOSITORY = "https://github.com/gubiczam/PROB.git"
PROB_BRANCH = "feat/daowod-bridge-v2"


class BridgeError(RuntimeError):
    """Raised when PROB is missing, unpatched, or a call fails."""


@dataclass
class Bridge:
    """A configured PROB checkout plus the dataset layout it reads."""

    prob_root: Path
    data_root: Path
    dataset: str = "OWDETR"
    device: str = "cuda"
    num_workers: int = 2
    log_dir: Path | None = None
    dry_run: bool = False
    calls: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.prob_root = Path(self.prob_root)
        self.data_root = Path(self.data_root)
        if self.log_dir is not None:
            self.log_dir = Path(self.log_dir)

    # ------------------------------------------------------------- setup ---

    @property
    def script(self) -> Path:
        return self.prob_root / "daowod_prob_bridge.py"

    def check(self) -> dict[str, object]:
        """Fail loudly and early rather than three hours into a chain."""

        if not self.script.exists():
            raise BridgeError(
                f"{self.script} is missing. Clone {PROB_REPOSITORY} at branch "
                f"{PROB_BRANCH}; the bridge only exists on that branch."
            )
        required = {
            "train": ("--labelled-ids", "--replay-ids", "--supervision-mode",
                      "--previous-checkpoint", "--output-checkpoint"),
            "predict": ("--image-ids", "--checkpoint", "--output"),
            "evaluate": ("--checkpoint", "--test-set", "--output"),
        }
        for command, flags in required.items():
            help_text = self._run_capture([sys.executable, str(self.script), command, "--help"])
            missing = [flag for flag in flags if flag not in help_text]
            if missing:
                raise BridgeError(f"PROB bridge '{command}' is missing {missing}.")
        return {
            "prob_root": str(self.prob_root),
            "data_root": str(self.data_root),
            "gpu": bool(shutil.which("nvidia-smi")),
            "verbs": sorted(required),
        }

    # -------------------------------------------------------------- verbs ---

    def predict(
        self,
        image_ids: Sequence[str],
        *,
        checkpoint: Path,
        output: Path,
        n_prev: int,
        n_current: int,
        max_proposals_per_image: int = 100,
    ) -> Path:
        """Export candidate regions for one round. Cached on ``output``."""

        output = Path(output)
        if output.exists():
            self._note("predict", output, cached=True)
            return output
        ids_file = self._write_ids(image_ids, output.with_name(output.stem + "_ids.txt"))
        self._call(
            "predict",
            ["--image-ids", str(ids_file),
             "--checkpoint", str(checkpoint),
             "--output", str(output),
             "--max-proposals-per-image", str(max_proposals_per_image)],
            n_prev=n_prev, n_current=n_current, label=f"predict:{output.stem}",
        )
        return output

    def train(
        self,
        labelled_ids: Sequence[str],
        *,
        previous_checkpoint: Path,
        output_checkpoint: Path,
        output_dir: Path,
        n_prev: int,
        n_current: int,
        replay_ids: Sequence[str] = (),
        supervision_mode: str = "ft",
        epochs: int = 5,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
    ) -> Path:
        """One incremental step. Cached on ``output_checkpoint``.

        ``supervision_mode='ft'`` keeps previous-task boxes that are already
        present in the selected images. The alternative, ``'train'``, drops them
        — which is what made forgetting look catastrophic in the earlier work,
        and is kept only so that comparison can be re-run.
        """

        output_checkpoint = Path(output_checkpoint)
        if output_checkpoint.exists():
            self._note("train", output_checkpoint, cached=True)
            return output_checkpoint
        ids_file = self._write_ids(labelled_ids, Path(output_dir) / "labelled_ids.txt")
        arguments = [
            "--labelled-ids", str(ids_file),
            "--previous-checkpoint", str(previous_checkpoint),
            "--output-checkpoint", str(output_checkpoint),
            "--output-dir", str(output_dir),
            "--supervision-mode", supervision_mode,
            "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
        ]
        if len(replay_ids):
            replay_file = self._write_ids(replay_ids, Path(output_dir) / "replay_ids.txt")
            arguments += ["--replay-ids", str(replay_file)]
        self._call("train", arguments, n_prev=n_prev, n_current=n_current,
                   label=f"train:{output_checkpoint.stem}")
        return output_checkpoint

    def evaluate(
        self,
        *,
        checkpoint: Path,
        test_set: str,
        output: Path,
        n_prev: int,
        n_current: int,
        batch_size: int = 4,
    ) -> Path:
        """PROB's official evaluator. Cached on ``output``."""

        output = Path(output)
        if output.exists():
            self._note("evaluate", output, cached=True)
            return output
        self._call(
            "evaluate",
            ["--checkpoint", str(checkpoint),
             "--test-set", test_set,
             "--output", str(output),
             "--output-dir", str(output.parent / f"{output.stem}_eval"),
             "--batch-size", str(batch_size),
             "--no-detections"],
            n_prev=n_prev, n_current=n_current, label=f"evaluate:{output.stem}",
        )
        return output

    # ----------------------------------------------------------- internals ---

    def _call(self, verb: str, arguments: list[str], *, n_prev: int, n_current: int, label: str) -> None:
        command = [
            sys.executable, str(self.script), verb,
            "--dataset", self.dataset,
            "--data-root", str(self.data_root),
            "--prev-introduced-classes", str(n_prev),
            "--current-introduced-classes", str(n_current),
            "--device", self.device,
            "--num-workers", str(self.num_workers),
            *arguments,
        ]
        if self.dry_run:
            self._note(verb, Path(label), cached=False, dry=True, command=command)
            return

        log_path = None
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(self.log_dir) / f"{label.replace(':', '_')}.log"

        started = time.time()
        handle = log_path.open("w", encoding="utf-8") if log_path else None
        try:
            process = subprocess.Popen(
                command, cwd=self.prob_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
            for line in process.stdout:
                if handle:
                    handle.write(line)
                if any(token in line for token in ("AP50", "Epoch:", "Error", "Traceback")):
                    print(line.rstrip()[:180], flush=True)
            code = process.wait()
        finally:
            if handle:
                handle.close()
        minutes = (time.time() - started) / 60
        self.calls.append({"verb": verb, "label": label, "minutes": minutes, "cached": False})
        if code != 0:
            raise BridgeError(
                f"PROB {verb} failed with exit code {code}."
                + (f" Log: {log_path}" if log_path else "")
            )
        print(f"  [{label}] {minutes:.1f} min", flush=True)

    def _note(self, verb: str, target: Path, *, cached: bool, dry: bool = False, command=None) -> None:
        self.calls.append(
            {"verb": verb, "label": target.stem, "minutes": 0.0, "cached": cached, "dry": dry}
        )
        if cached:
            print(f"  [{verb}] cached: {target.name}", flush=True)
        elif dry:
            print(f"  [{verb}] dry run: {' '.join(str(x) for x in (command or [])[-6:])}", flush=True)

    @staticmethod
    def _write_ids(image_ids: Sequence[str], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(value) for value in image_ids) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _run_capture(command: list[str]) -> str:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise BridgeError(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def cost_report(self) -> dict[str, float]:
        """Minutes spent per verb, so a chain can be priced before it is run."""

        report: dict[str, float] = {}
        for call in self.calls:
            report[call["verb"]] = report.get(call["verb"], 0.0) + float(call["minutes"])
        report["total"] = sum(report.values())
        return report


def ensure_checkout(root: Path, *, repository: str = PROB_REPOSITORY, branch: str = PROB_BRANCH) -> Path:
    """Clone or update PROB. Colab-only; a no-op if the checkout is already there."""

    root = Path(root)
    if root.exists():
        subprocess.run(["git", "fetch", "--depth", "1", "origin", branch], cwd=root, check=True)
        subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=root, check=True)
    else:
        subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1", repository, str(root)],
            check=True,
        )
    return root


def read_metrics(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
