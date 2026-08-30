from reasoning.generate_submission import EXPECTED_KEYS, ROLLOUTS_PER_KEY, validate_submission


def test_validate_submission_accepts_exact_challenge_shape():
    validate_submission({f"clip_{i}": [f"reasoning {j}" for j in range(ROLLOUTS_PER_KEY)] for i in range(EXPECTED_KEYS)})


def test_validate_submission_rejects_missing_event():
    payload = {f"clip_{i}": ["x"] * ROLLOUTS_PER_KEY for i in range(EXPECTED_KEYS - 1)}
    try:
        validate_submission(payload)
    except ValueError as exc:
        assert "expected 284 keys" in str(exc)
    else:
        raise AssertionError("missing event was accepted")
