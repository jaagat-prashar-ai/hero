from rl_posttrain.rewards.code_reward_entry import _two_tier_gate_confidence


def test_two_tier_rejects_sample_failure():
    assert _two_tier_gate_confidence({}, {"sample_failures": ["no valid rollout"]}) == 0.0


def test_two_tier_rejects_nonfinite_reward():
    assert _two_tier_gate_confidence({}, {"reward_failures": ["only 4/12 rollouts scored finitely"]}) == 0.0


def test_two_tier_keeps_directionally_useful_ranking():
    metrics = {
        "code_group_gate_score_range": 0.12,
        "code_group_reward_consistency_corr": 0.24,
        "code_group_argmax_consistency_lift": 0.04,
        "code_group_gate_min_delta": 0.20,
    }
    assert _two_tier_gate_confidence(metrics, {"sample_failures": [], "reward_failures": []}) > 0.0


def test_two_tier_rejects_uninformative_ranking():
    metrics = {
        "code_group_gate_score_range": 0.01,
        "code_group_reward_consistency_corr": -0.2,
        "code_group_argmax_consistency_lift": -0.1,
        "code_group_gate_min_delta": 0.0,
    }
    assert _two_tier_gate_confidence(metrics, {"sample_failures": [], "reward_failures": []}) == 0.0
