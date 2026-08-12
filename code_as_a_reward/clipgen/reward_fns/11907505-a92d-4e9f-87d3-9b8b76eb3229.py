"""clip 11907505-a92d-4e9f-87d3-9b8b76eb3229 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "perceptual_lead_vehicle": 0.0,
        "commitment_keep_distance": 0.0,
        "conjunction_deceleration_and_claim": 0.0,
        "trajectory_lateral_stability": 0.0,
        "trajectory_heading_stability": 0.0
    }

    # Check perceptual claims
    lead_vehicle_claim = any(pc.entity == "lead_vehicle" and pc.state == "ahead" for pc in claims.perceptual)
    if lead_vehicle_claim:
        scores["perceptual_lead_vehicle"] = 0.10

    # Check commitment claims
    keep_distance_commitment = any(cc.maneuver == "keep_distance" for cc in claims.commitments)
    if keep_distance_commitment:
        scores["commitment_keep_distance"] = 0.10

    # Check trajectory for deceleration and require claim
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps

        # Ensure significant speed drop occurs by the end of the window
        if initial_speed - min_speed >= 5.0 and min_speed <= 19.0 and np.argmin(speed_window) >= 50:
            if lead_vehicle_claim and keep_distance_commitment:
                scores["conjunction_deceleration_and_claim"] = 0.50

    # Check trajectory for lateral stability
    if abs(traj.final_lateral_offset_m) <= 0.5:
        scores["trajectory_lateral_stability"] = 0.15

    # Check trajectory for heading stability
    if abs(traj.total_heading_change_deg) <= 1.0:
        scores["trajectory_heading_stability"] = 0.15

    return scores

def reward(claims, traj):
    """Reward function for scene with decisive event: Deceleration to maintain safe distance from lead vehicle.
    Thresholds derived from dossier: speed drop >= 5.0 m/s, min speed <= 19.0 m/s, lateral offset <= 0.5 m, heading change <= 1.0 deg."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
