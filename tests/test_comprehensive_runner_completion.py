from argparse import Namespace
import json

from scripts.evaluation.run_comprehensive_evaluation import completed_outer_run


def arguments():
    return Namespace(rounds=5, validation_repeats=10)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_alone_is_not_complete(tmp_path):
    write_json(tmp_path / "results_summary.json", {"num_rounds_completed": 5})
    complete, reason = completed_outer_run(tmp_path, arguments())
    assert not complete
    assert reason.startswith("missing:")


def test_complete_run_requires_selected_validation_repeats(tmp_path):
    selection = {
        "selected": {"validation_result_key": "shortlist_rank_01"}
    }
    write_json(
        tmp_path / "results_summary.json",
        {"num_rounds_completed": 5, "independent_selection": selection},
    )
    write_json(
        tmp_path / "independent_validation.json",
        {
            "selection": selection,
            "shortlist_rank_01": {"perceptual_evaluations": [{}] * 10},
        },
    )
    write_json(tmp_path / "selected_validated_profile.json", selection["selected"])
    assert completed_outer_run(tmp_path, arguments()) == (True, "complete")


def test_incomplete_validation_is_not_skipped(tmp_path):
    selection = {
        "selected": {"validation_result_key": "shortlist_rank_01"}
    }
    write_json(
        tmp_path / "results_summary.json",
        {"num_rounds_completed": 5, "independent_selection": selection},
    )
    write_json(
        tmp_path / "independent_validation.json",
        {
            "selection": selection,
            "shortlist_rank_01": {"perceptual_evaluations": [{}] * 4},
        },
    )
    write_json(tmp_path / "selected_validated_profile.json", selection["selected"])
    complete, reason = completed_outer_run(tmp_path, arguments())
    assert not complete
    assert reason == "selected_validation_repeat_mismatch"
