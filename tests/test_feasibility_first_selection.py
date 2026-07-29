from laban_rl.perceptual_bandit.selection import select_feasible_incumbent


def result(reward, *, rmse=0.01, max_error=0.02, physical=True):
    return {
        "requested_profile": {"weight": 0.5},
        "outer_reward": reward,
        "valid_realisation": True,
        "physically_acceptable": physical,
        "realisation_rmse": rmse,
        "max_abs_feature_error": max_error,
    }


def shortlist_item(rank, key, reward):
    return {
        "rank": rank,
        "source": f"sample_{rank}",
        "training_reward": reward,
        "profile": {"weight": 0.4 + rank / 10},
        "validation_reward": reward,
        "validation_result_key": key,
    }


def test_infeasible_high_reward_cannot_beat_feasible_candidate():
    validation = {
        "initial_profile": result(0.20),
        "final_distribution_mean": result(0.30),
        "shortlist_rank_01": result(0.90, max_error=0.25),
    }
    selected = select_feasible_incumbent(
        validation,
        [shortlist_item(1, "shortlist_rank_01", 0.90)],
        tolerance=0.10,
    )
    assert selected["selected"]["source"] == "final_distribution_mean"
    assert selected["strict_feasibility_satisfied"]


def test_feasible_initial_profile_prevents_regression():
    validation = {
        "initial_profile": result(0.50),
        "final_distribution_mean": result(0.10),
        "shortlist_rank_01": result(0.40),
    }
    selected = select_feasible_incumbent(
        validation,
        [shortlist_item(1, "shortlist_rank_01", 0.40)],
        tolerance=0.10,
    )
    assert selected["selected"]["source_type"] == "incumbent_initial"
    assert selected["selected"]["validation_reward"] == 0.50

