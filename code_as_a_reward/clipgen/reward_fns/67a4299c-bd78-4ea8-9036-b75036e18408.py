"""clip 67a4299c-bd78-4ea8-9036-b75036e18408 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 8)"""
def components(claims, traj):
    """Components for decisive events: navigating construction zone and maintaining safe distance from lead vehicle.
    Scene-derived thresholds:
    - Lateral offset: max |1.31 m|, final -1.31 m
    - Speed: initial 9.3 m/s, final 5.8 m/s, min 5.7 m/s (drop 3.5 m/s)
    - Total heading change: -2 degrees
    """

    # Initialize component scores
    comp = {
        "perceive_lead_vehicle": 0.0,
        "maintain_safe_distance": 0.0
    }

    # Check for perceptual claims
    if any(claim.entity == 'lead_vehicle' for claim in claims.perceptual):
        comp["perceive_lead_vehicle"] = 0.2

    # Check for commitment claims and trajectory execution
    if any(claim.maneuver == 'keep_distance' for claim in claims.commitments):
        # Check trajectory for maintaining safe distance
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed = traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.dt_s * traj.n_waypoints))

        if 3.0 <= speed_drop <= 4.0 and min_speed >= 5.0 and min_speed_time >= 50:
            comp["maintain_safe_distance"] = 0.5

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
