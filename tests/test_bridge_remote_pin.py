"""Verifying the pinned PROB remote before an overnight run spends anything.

A Method V3 run died in the PROB setup cell on

    git clone --filter=blob:none --no-checkout .../PROB.git  ->  exit status 128

after Drive was mounted and OWL installed. A bare 128 does not distinguish a
wrong URL from a deleted commit from a rate-limited egress address, and the
tempting repair — point at some other PROB — would silently replace the frozen
detector. So the question is answered before the expensive setup, and these tests
pin what "answered" means.

No test here touches the network: ``verify_remote_commit`` takes its two probes
as parameters.
"""

from __future__ import annotations

import pytest

from owl import bridge

PIN = "4c66be1a52cad9360e09c729e9134aba8fe0b531"
OTHER = "cbd5bfd355f4b97ba6a3aa8769c9d2b428385b39"

#: What the real server answered on 2026-09-03, recorded so the shape of the
#: reply this code parses is the shape it was written against.
REAL_LS_REMOTE = (
    f"874c055310943ab749942339e8096a4843fc4d47\trefs/heads/feat/daowod-bridge\n"
    f"{PIN}\trefs/heads/feat/daowod-bridge-v2\n"
    f"{OTHER}\trefs/heads/main\n"
)


def reachable(output=REAL_LS_REMOTE):
    def probe(repository, timeout):
        return 0, output, ""
    return probe


def unreachable(code=128, error="fatal: repository not found"):
    calls = []

    def probe(repository, timeout):
        calls.append(repository)
        return code, "", error

    probe.calls = calls
    return probe


def never_fetchable(repository, commit, timeout):
    return False, "fatal: could not fetch that object"


def always_fetchable(repository, commit, timeout):
    return True, ""


# ------------------------------------------------------------- the pin holds ---


def test_the_projects_pin_is_recognised_as_a_branch_tip():
    report = bridge.verify_remote_commit(
        bridge.PROB_REPOSITORY, PIN,
        ls_remote=reachable(), fetchable=never_fetchable,
    )
    assert report["pin_is_ref_tip"] is True
    assert report["refs_at_commit"] == ["refs/heads/feat/daowod-bridge-v2"]
    assert report["branch_points_at_commit"] is True
    assert report["branch_head"] == PIN
    assert report["attempts_used"] == 1


def test_the_frozen_url_and_branch_are_the_projects_own_fork():
    """Recorded so a change of detector source cannot pass unnoticed."""

    assert bridge.PROB_REPOSITORY == "https://github.com/gubiczam/PROB.git"
    assert bridge.PROB_BRANCH == "feat/daowod-bridge-v2"


def test_a_moved_branch_still_honours_the_pin():
    """A pin that has become an interior commit is a pin, not an error."""

    moved = REAL_LS_REMOTE.replace(
        f"{PIN}\trefs/heads/feat/daowod-bridge-v2", f"{OTHER}\trefs/heads/feat/daowod-bridge-v2"
    )
    report = bridge.verify_remote_commit(
        bridge.PROB_REPOSITORY, PIN,
        ls_remote=reachable(moved), fetchable=always_fetchable,
    )
    assert report["pin_is_ref_tip"] is False
    assert report["branch_points_at_commit"] is False
    assert report["branch_head"] == OTHER


# ------------------------------------------------------------- it fails loud ---


def test_an_unreachable_url_names_the_url_and_the_sha():
    probe = unreachable()
    with pytest.raises(bridge.BridgeError) as raised:
        bridge.verify_remote_commit(
            bridge.PROB_REPOSITORY, PIN,
            attempts=2, delay=0.0, ls_remote=probe, fetchable=never_fetchable,
        )
    message = str(raised.value)
    assert bridge.PROB_REPOSITORY in message
    assert PIN in message
    assert bridge.PROB_BRANCH in message
    assert "repository not found" in message
    assert "exit 128" in message


def test_an_unreachable_url_is_retried_before_it_is_believed():
    """The observed failure was transient; one dropped probe is not a verdict."""

    probe = unreachable()
    with pytest.raises(bridge.BridgeError):
        bridge.verify_remote_commit(
            bridge.PROB_REPOSITORY, PIN,
            attempts=3, delay=0.0, ls_remote=probe, fetchable=never_fetchable,
        )
    assert len(probe.calls) == 3


def test_a_transient_failure_that_clears_is_not_an_error():
    outcomes = [(128, "", "fatal: the remote end hung up"), (0, REAL_LS_REMOTE, "")]

    def probe(repository, timeout):
        return outcomes.pop(0)

    report = bridge.verify_remote_commit(
        bridge.PROB_REPOSITORY, PIN,
        attempts=3, delay=0.0, ls_remote=probe, fetchable=never_fetchable,
    )
    assert report["attempts_used"] == 2
    assert report["pin_is_ref_tip"] is True


def test_a_missing_commit_fails_and_lists_what_the_server_did_offer():
    without = REAL_LS_REMOTE.replace(
        f"{PIN}\trefs/heads/feat/daowod-bridge-v2\n", ""
    )
    with pytest.raises(bridge.BridgeError) as raised:
        bridge.verify_remote_commit(
            bridge.PROB_REPOSITORY, PIN,
            ls_remote=reachable(without), fetchable=never_fetchable,
        )
    message = str(raised.value)
    assert "not on that server" in message
    assert PIN in message
    assert "refs/heads/main" in message


def test_the_error_never_proposes_another_repository():
    """The whole point: a convenient fork is not an acceptable repair."""

    for probe, fetch in ((unreachable(), never_fetchable),
                         (reachable(REAL_LS_REMOTE.replace(PIN, OTHER)),
                          never_fetchable)):
        with pytest.raises(bridge.BridgeError) as raised:
            bridge.verify_remote_commit(
                bridge.PROB_REPOSITORY, PIN,
                attempts=1, delay=0.0, ls_remote=probe, fetchable=fetch,
            )
        message = str(raised.value)
        assert "orrzohar" not in message
        assert "latest" not in message.lower()
        assert message.count("github.com") <= 2, "no alternative URL is offered"


def test_the_pin_must_be_a_full_sha():
    for bad in ("4c66be1", "", "main", "Z" * 40):
        with pytest.raises(bridge.BridgeError, match="not a full 40-character"):
            bridge.verify_remote_commit(
                bridge.PROB_REPOSITORY, bad,
                ls_remote=reachable(), fetchable=never_fetchable,
            )


def test_an_uppercase_pin_is_accepted_and_normalised():
    report = bridge.verify_remote_commit(
        bridge.PROB_REPOSITORY, PIN.upper(),
        ls_remote=reachable(), fetchable=never_fetchable,
    )
    assert report["commit"] == PIN


# --------------------------------------------------- the offline escape hatch ---


def test_url_normalisation_treats_the_equivalent_forms_as_one():
    forms = (
        "https://github.com/gubiczam/PROB.git",
        "https://github.com/gubiczam/PROB",
        "https://github.com/gubiczam/PROB/",
        "git@github.com:gubiczam/PROB.git",
    )
    assert len({bridge.normalise_repository(form) for form in forms}) == 1


def test_an_exact_local_checkout_is_accepted(tmp_path):
    (tmp_path / ".git").mkdir()
    answers = {("remote", "get-url", "origin"): (0, "git@github.com:gubiczam/PROB.git"),
               ("rev-parse", "HEAD"): (0, PIN.upper())}
    assert bridge.local_checkout_matches(
        tmp_path, bridge.PROB_REPOSITORY, PIN,
        runner=lambda path, arguments: answers[arguments],
    )


def test_a_checkout_at_the_wrong_commit_is_refused(tmp_path):
    (tmp_path / ".git").mkdir()
    answers = {("remote", "get-url", "origin"): (0, bridge.PROB_REPOSITORY),
               ("rev-parse", "HEAD"): (0, OTHER)}
    assert not bridge.local_checkout_matches(
        tmp_path, bridge.PROB_REPOSITORY, PIN,
        runner=lambda path, arguments: answers[arguments],
    )


def test_a_checkout_of_a_different_repository_is_refused(tmp_path):
    (tmp_path / ".git").mkdir()
    answers = {("remote", "get-url", "origin"):
               (0, "https://github.com/orrzohar/PROB.git"),
               ("rev-parse", "HEAD"): (0, PIN)}
    assert not bridge.local_checkout_matches(
        tmp_path, bridge.PROB_REPOSITORY, PIN,
        runner=lambda path, arguments: answers[arguments],
    )


def test_a_path_that_is_not_a_git_checkout_is_refused(tmp_path):
    assert not bridge.local_checkout_matches(
        tmp_path, bridge.PROB_REPOSITORY, PIN,
        runner=lambda path, arguments: (0, PIN),
    )
