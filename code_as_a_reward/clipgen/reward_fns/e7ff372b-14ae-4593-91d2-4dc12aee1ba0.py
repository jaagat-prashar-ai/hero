"""clip e7ff372b-14ae-4593-91d2-4dc12aee1ba0 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "committed_and_executed_yield": 0.0,
    }

    # Check for commitment to yield and execution
    committed_to_yield = any(
        claim.maneuver == "yield" and claim.speed_profile == "decelerate"
        for claim in claims.commitments
    )

    if committed_to_yield and traj.n_waypoints > 0:
        # Calculate speed drop and timing within the rollout horizon
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s

        # Check if the speed drop is significant and occurs at the correct time
        if speed_drop >= 5.0 and 5.0 <= min_speed_time <= 6.4:
            scores["committed_and_executed_yield"] = 0.7

    return scores

def reward(claims, traj):
    """Reward function for yielding to a pedestrian crossing:
    - Commits to yield and executes significant speed reduction (>= 5.0 m/s) within the correct timing window
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
