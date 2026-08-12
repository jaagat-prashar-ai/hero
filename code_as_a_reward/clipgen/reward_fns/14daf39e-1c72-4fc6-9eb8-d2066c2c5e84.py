"""clip 14daf39e-1c72-4fc6-9eb8-d2066c2c5e84 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "committed_and_executed_yield": 0.0
    }

    # Check for commitment to yield and executed yield (conjunction)
    committed_to_yield = any(
        claim.maneuver == "yield" and claim.speed_profile == "decelerate"
        for claim in claims.commitments
    )
    if committed_to_yield and traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed = min(speed_window) if len(speed_window) > 0 else initial_speed
        speed_drop = initial_speed - min_speed

        # Consider a speed drop of at least 5 m/s as acceptable
        if speed_drop >= 5.0:
            scores["committed_and_executed_yield"] = 0.8

    return scores

def reward(claims, traj):
    """Reward function for scene with decisive event: Yield to pedestrian crossing.
    Thresholds: speed drop >= 5 m/s."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
