"""clip 5a4dfdb5-65f4-40d1-b1d7-60db3bd8b227 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 3)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "committed_to_yield_and_executed": 0.0
    }

    # Check for commitment to yield
    committed_yield = any(
        claim.maneuver == "yield" and claim.speed_profile == "decelerate"
        for claim in claims.commitments
    )

    # Check trajectory for execution of yield (deceleration)
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        min_speed_time = np.argmin(speed_window) * traj.dt_s

        # Check if speed drops significantly and at the correct time
        if committed_yield and initial_speed - min_speed >= 0.5 and 1.5 <= min_speed_time <= 3.0:
            scores["committed_to_yield_and_executed"] = 0.7

    return scores

def reward(claims, traj):
    """Reward function for yielding to a cyclist crossing the street.
    - Commit to yield and execute yield (deceleration): 0.7
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
