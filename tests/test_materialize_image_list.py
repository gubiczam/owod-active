"""The materialiser's ``--image-list`` mode: parsing, reuse, and failing closed.

Infrastructure only. No network in these tests: :func:`fetch` is replaced, so
what is exercised is the parsing, the deduplication, the already-present path and
the failure behaviour -- not COCO's availability.

The failure behaviour is the point. An export gated on "every image present" is
only as good as the fetch that fed it, so a partial materialisation must exit
non-zero rather than leave a root that looks ready and fails later, further from
its cause.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import materialize_pool_images as materialise


def _jpeg(path: Path, image_id: str) -> Path:
    """A minimal real JPEG, so ``valid_jpeg`` genuinely verifies it."""

    from PIL import Image

    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{image_id}.jpg"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(target, format="JPEG")
    return target


# ------------------------------------------------------------------ parsing ---


def test_ids_are_stripped_and_blank_lines_ignored(tmp_path):
    listing = tmp_path / "ids.txt"
    listing.write_text(
        "  000000000025  \n\n000000000034\n\t000000000036\t\n\n", encoding="utf-8")

    assert materialise.parse_image_list(listing) == [
        "000000000025", "000000000034", "000000000036",
    ]


def test_duplicates_are_dropped_keeping_the_first_occurrence(tmp_path):
    listing = tmp_path / "ids.txt"
    listing.write_text(
        "000000000034\n000000000025\n000000000034\n000000000025\n", encoding="utf-8")

    # order is a function of the file alone, so the fetch order is reproducible
    assert materialise.parse_image_list(listing) == ["000000000034", "000000000025"]


def test_parsing_is_deterministic_across_calls(tmp_path):
    listing = tmp_path / "ids.txt"
    listing.write_text("\n".join(f"{i:012d}" for i in range(50)), encoding="utf-8")

    assert materialise.parse_image_list(listing) == materialise.parse_image_list(listing)


@pytest.mark.parametrize(
    "bad",
    [
        "25",                    # not zero-padded
        "0000000000251",         # too long
        "00000000002a",          # not numeric
        "../../etc/passwd",      # would escape JPEGImages
        "000000000025.jpg",      # extension included
    ],
)
def test_malformed_ids_are_refused(tmp_path, bad):
    listing = tmp_path / "ids.txt"
    listing.write_text(f"000000000034\n{bad}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="malformed image id"):
        materialise.parse_image_list(listing)


def test_all_malformed_ids_are_reported_together(tmp_path):
    """One round trip, not one per bad line."""

    listing = tmp_path / "ids.txt"
    listing.write_text("bad1\nbad2\nbad3\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        materialise.parse_image_list(listing)
    assert "3 malformed" in str(error.value)


def test_an_empty_or_missing_list_is_refused(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n \n\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no image ids"):
        materialise.parse_image_list(empty)
    with pytest.raises(SystemExit, match="does not exist"):
        materialise.parse_image_list(tmp_path / "absent.txt")


def test_valid_image_id_accepts_only_zero_padded_twelve_digits():
    assert materialise.valid_image_id("000000000025")
    assert not materialise.valid_image_id("00000000025")
    assert not materialise.valid_image_id("00000000002x")


# -------------------------------------------------------------- fetch logic ---


def test_already_present_images_are_counted_and_not_refetched(tmp_path, monkeypatch):
    jpeg = tmp_path / "JPEGImages"
    _jpeg(jpeg, "000000000025")
    called: list[str] = []
    monkeypatch.setattr(materialise, "fetch",
                        lambda name, target: (called.append(name), (name, None))[1])

    counts = materialise.materialise(
        ["000000000025", "000000000034"], jpeg, workers=2)

    assert called == ["000000000034"], "a valid existing JPEG must not be refetched"
    assert counts["already_present"] == 1
    assert counts["requested"] == 2


def test_a_valid_existing_jpeg_is_left_byte_identical(tmp_path, monkeypatch):
    jpeg = tmp_path / "JPEGImages"
    target = _jpeg(jpeg, "000000000025")
    before = target.read_bytes()
    monkeypatch.setattr(materialise, "fetch", lambda name, t: (name, None))

    materialise.materialise(["000000000025"], jpeg, workers=1)

    assert target.read_bytes() == before


def test_counts_add_up_over_present_and_downloaded(tmp_path, monkeypatch):
    jpeg = tmp_path / "JPEGImages"
    _jpeg(jpeg, "000000000025")

    def fake_fetch(name, target):
        _jpeg(target, name)
        return name, None

    monkeypatch.setattr(materialise, "fetch", fake_fetch)

    counts = materialise.materialise(
        ["000000000025", "000000000034", "000000000036", "000000000034"],
        jpeg, workers=2)

    assert counts["requested"] == 4          # duplicates included in "requested"
    assert counts["unique"] == 3
    assert counts["already_present"] == 1
    assert counts["downloaded"] == 2
    assert counts["failed"] == 0


def test_the_result_is_validated_not_the_return_code(tmp_path, monkeypatch):
    """A fetch that claims success but writes nothing must still be a failure."""

    jpeg = tmp_path / "JPEGImages"
    monkeypatch.setattr(materialise, "fetch", lambda name, target: (name, None))

    counts = materialise.materialise(["000000000025"], jpeg, workers=1)

    assert counts["failed"] == 1
    assert counts["unreadable"] == ["000000000025"]


def test_a_zero_byte_result_is_treated_as_missing(tmp_path, monkeypatch):
    jpeg = tmp_path / "JPEGImages"
    jpeg.mkdir(parents=True)

    def empty_fetch(name, target):
        (target / f"{name}.jpg").write_bytes(b"")
        return name, None

    monkeypatch.setattr(materialise, "fetch", empty_fetch)

    assert materialise.materialise(["000000000025"], jpeg, workers=1)["failed"] == 1


def test_a_non_jpeg_result_is_treated_as_missing(tmp_path, monkeypatch):
    jpeg = tmp_path / "JPEGImages"
    jpeg.mkdir(parents=True)

    def html_fetch(name, target):
        (target / f"{name}.jpg").write_bytes(b"<html>404</html>")
        return name, None

    monkeypatch.setattr(materialise, "fetch", html_fetch)

    assert materialise.materialise(["000000000025"], jpeg, workers=1)["failed"] == 1


# --------------------------------------------------------------- fail closed ---


def test_report_exits_non_zero_when_anything_is_missing(tmp_path, capsys):
    counts = {"requested": 3, "unique": 3, "already_present": 1, "downloaded": 1,
              "failed": 1, "failures": [("000000000036", "404")],
              "unreadable": ["000000000036"]}

    with pytest.raises(SystemExit) as error:
        materialise.report(counts, tmp_path / "JPEGImages")

    message = str(error.value)
    assert "1 of 3 images could not be materialised" in message
    assert "000000000036" in message


def test_report_prints_every_required_count(tmp_path, capsys):
    counts = {"requested": 4, "unique": 3, "already_present": 1, "downloaded": 2,
              "failed": 0, "failures": [], "unreadable": []}

    materialise.report(counts, tmp_path / "JPEGImages")

    printed = capsys.readouterr().out
    for label in ("requested", "unique", "already present", "downloaded", "failed"):
        assert label in printed
    assert "4" in printed and "3" in printed


def test_image_list_mode_writes_the_canonical_layout(tmp_path, monkeypatch):
    """The same <data-root>/JPEGImages/<id>.jpg the exporters read."""

    listing = tmp_path / "ids.txt"
    listing.write_text("000000000025\n000000000034\n", encoding="utf-8")
    root = tmp_path / "OWOD"

    def fake_fetch(name, target):
        _jpeg(target, name)
        return name, None

    monkeypatch.setattr(materialise, "fetch", fake_fetch)
    monkeypatch.setattr(
        "sys.argv",
        ["materialize_pool_images.py", "--data-root", str(root),
         "--image-list", str(listing)],
    )

    materialise.main()

    for image_id in ("000000000025", "000000000034"):
        assert (root / "JPEGImages" / f"{image_id}.jpg").is_file()
    # image-list mode fetches JPEGs only: no archive, no ImageSets file
    assert not (root / "Annotations").exists()
    assert not (root / "ImageSets").exists()
