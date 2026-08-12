"""clip 16638d32-3382-4f83-be1a-4ebd8f893767 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.20, real rollout argmax 8)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive events of navigating
    through a construction zone while following a lead vehicle. The thresholds are
    derived from the ground-truth dossier:
    - Speed should decrease slightly, with a tolerance of ±0.5 m/s.
    - Total heading change should be within ±2 degrees of -5.2 degrees.
    """
    # Initialize component scores
    comp = {
        "perceived_lead_vehicle_and_commitment": 0.0,
        "commitment_and_speed_profile": 0.0,
        "heading_change_match": 0.0
    }

    # Check perceptual claims and commitment
    if any(pc.entity == 'lead_vehicle' for pc in claims.perceptual) and \
       any(cc.maneuver == 'accelerate' for cc in claims.commitments):
        comp["perceived_lead_vehicle_and_commitment"] = 0.2

    # Check commitment claims and trajectory execution
    if any(cc.maneuver == 'accelerate' and cc.speed_profile == 'accelerate' for cc in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed_time = np.argmin(traj.speed_mps) * traj.dt_s
        if 0.3 <= speed_drop <= 1.3 and 5.0 <= min_speed_time <= 7.0:
            comp["commitment_and_speed_profile"] = 0.4

    # Check heading change
    if any(cc.maneuver == 'accelerate' for cc in claims.commitments) and \
       -7.2 <= traj.total_heading_change_deg <= -3.2:  # Allowing a tolerance of ±2 degrees
        comp["heading_change_match"] = 0.2

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
