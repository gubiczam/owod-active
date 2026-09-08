"""The figure generator: every panel traceable, and nothing invented.

A slide is where a number gets separated from its provenance, so these pin the
two properties that matter: each panel names the committed file it came from,
and the one panel whose data is not in this repository is *declared and skipped*
rather than drawn from remembered values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "build_consultation_figures.py"

pytest.importorskip("matplotlib")


@pytest.fixture(scope="module")
def built(tmp_path_factory, capsys_disabled=None):
    from tools.build_consultation_figures import main

    out = tmp_path_factory.mktemp("figures")
    assert main(["--out", str(out)]) == 0
    return out


def test_every_committed_panel_is_produced(built):
    names = sorted(p.stem for p in built.glob("*.png"))
    assert names == [
        "fig1_forgetting_decomposition",
        "fig2_cost_is_not_supervision",
        "fig3_acquisition_vs_learning",
        "fig4_coherence_gate",
        "fig5_batch_diversity",
        "fig6_rounds",
    ], names
    for path in built.glob("*.png"):
        assert path.stat().st_size > 10_000, f"{path.name} looks empty"


def test_the_benchmark_panel_is_skipped_not_invented(built, capsys):
    """It needs a file that lives on Drive. It must never be drawn anyway."""

    from tools.build_consultation_figures import main

    main(["--out", str(built)])
    printed = capsys.readouterr().out
    assert "fig7_three_seed_new_ap.png  SKIPPED" in printed
    assert "NOT drawn from remembered numbers" in printed
    assert not (built / "fig7_three_seed_new_ap.png").exists()


def test_every_panel_reports_its_source_and_a_caveat(built, capsys):
    from tools.build_consultation_figures import main

    main(["--out", str(built)])
    printed = capsys.readouterr().out
    assert printed.count("source :") == 6
    assert printed.count("caveat :") == 6
    for token in ("real_group_forgetting.csv", "labelling_policy.csv",
                  "coherence_gate.csv", "batch_diversity_validation.csv",
                  "selection_arms.csv"):
        assert token in printed, token


def test_the_tool_reads_only_committed_files():
    source = TOOL.read_text(encoding="utf-8")
    for path in ("data/reference/measured/real_group_forgetting.csv",
                 "data/results/labelling_policy.csv"):
        assert (ROOT / path).is_file(), path
    # No network and no Drive *access* -- the word appears in the skip message
    # that explains why one panel is not drawn, which is the point of it.
    for forbidden in ("import requests", "import urllib", "/content/drive",
                      "urlopen", "subprocess"):
        assert forbidden not in source, forbidden


def test_the_caveats_say_what_the_numbers_are_not(built, capsys):
    """The captions must carry the scope limits, not only the values.

    Checked on the *rendered* caveats rather than on the source, because that
    is what reaches a slide -- and because a caveat split across two source
    lines is still one sentence to the reader.
    """

    from tools.build_consultation_figures import main

    main(["--out", str(built)])
    printed = " ".join(capsys.readouterr().out.split())
    for scope in ("predecessor protocol",
                  "known_plus_selected has never been run on the detector",
                  "PROB decoder features only",
                  "Never run downstream on the detector",
                  "hide a real disagreement"):
        assert scope in printed, scope
