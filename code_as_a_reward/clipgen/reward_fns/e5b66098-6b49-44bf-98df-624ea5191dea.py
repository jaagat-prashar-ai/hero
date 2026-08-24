"""clip e5b66098-6b49-44bf-98df-624ea5191dea - attempt 4/5 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene:
    - Recognize nearby vehicles and decelerate in response.
    - Maintain lane stability with minimal lateral movement.
    - Scene-derived thresholds: speed drop >= 2.15 m/s, lateral offset within ±0.17 m.
    """
    comp = {
        "decelerate_commitment": 0.0,
        "speed_reduction": 0.0,
        "lateral_stability": 0.0
    }

    # Commitment component: deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory component: speed reduction
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if speed_drop >= 2.15 and min_speed_time >= 3.0:
            comp["decelerate_commitment"] = 0.3
            comp["speed_reduction"] = 0.5 * min(1.0, speed_drop / 4.3)

    # Lateral stability component: requires both commitment and trajectory
    if any(c.speed_profile == 'decelerate' for c in claims.commitments) and abs(traj.final_lateral_offset_m) <= 0.17:
        comp["lateral_stability"] = 0.2

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
