"""clip 8505b3db-fb6b-49b6-a23a-5882a6dc1780 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Gentle deceleration to maintain a safe distance from the lead vehicle.
    
    Scene-derived thresholds:
    - Speed drop: approximately 4.9 m/s over 6.4 seconds.
    - Minimal heading change: approximately -4 degrees.
    - Perceptual and commitment claims must align with trajectory actions.
    """

    # Initialize component scores
    components = {
        "perceptual_lead_vehicle": 0.0,
        "commitment_decelerate": 0.0,
        "commitment_keep_distance": 0.0,
        "trajectory_deceleration": 0.0
    }

    # Check perceptual claims
    if any(pc.entity == 'lead_vehicle' and pc.state == 'ahead' for pc in claims.perceptual):
        components["perceptual_lead_vehicle"] = 0.1

    # Check commitment claims
    decelerate_claim = any(cc.maneuver == 'decelerate' and cc.speed_profile == 'decelerate' for cc in claims.commitments)
    keep_distance_claim = any(cc.maneuver == 'keep_distance' and cc.speed_profile == 'maintain' for cc in claims.commitments)

    if decelerate_claim:
        components["commitment_decelerate"] = 0.2

    if keep_distance_claim:
        components["commitment_keep_distance"] = 0.2

    # Check trajectory for deceleration
    speed_drop = traj.initial_speed_mps - traj.final_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
    if 4.5 <= speed_drop <= 5.3 and min_speed_time == traj.n_waypoints - 1:  # Ensure the min speed occurs at the end
        if decelerate_claim:
            components["trajectory_deceleration"] = 0.5

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
