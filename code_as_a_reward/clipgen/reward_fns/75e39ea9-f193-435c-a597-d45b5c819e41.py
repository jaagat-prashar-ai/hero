"""clip 75e39ea9-f193-435c-a597-d45b5c819e41 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to cyclists
    while maintaining speed. The thresholds are inspired by the ground-truth trajectory
    and reasoning, allowing for reasonable deviations in a real rollout.
    """
    # Initialize component scores
    scores = {
        "saw_cyclists": 0.0,
        "committed_to_yield": 0.0,
        "executed_yield": 0.0,
        "maintained_speed": 0.0,
        "lateral_control": 0.0
    }

    # Check perceptual claims
    saw_cyclists = any(pc.entity == 'cyclist' and pc.state == 'crossing' for pc in claims.perceptual)
    if saw_cyclists:
        scores["saw_cyclists"] = 0.1

    # Check commitment claims
    committed_to_yield = any(cc.maneuver == 'yield' for cc in claims.commitments)
    if committed_to_yield:
        scores["committed_to_yield"] = 0.1

    # Trajectory checks
    if traj.n_waypoints > 0:
        # Speed management: check for consistent speed increase
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps

        # Check if speed increases consistently and requires a yield commitment
        if committed_to_yield and final_speed > initial_speed and len(speed_window) > 0 and np.all(np.diff(speed_window) > -0.1):
            scores["executed_yield"] = 0.4

        # Lateral control: ensure minimal lateral offset and requires a yield commitment
        lateral_offset_window = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
        final_lateral_offset = traj.final_lateral_offset_m
        if committed_to_yield and abs(final_lateral_offset) <= 0.1:
            scores["lateral_control"] = 0.2

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
