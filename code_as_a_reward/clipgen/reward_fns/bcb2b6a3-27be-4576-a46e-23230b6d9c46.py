"""clip bcb2b6a3-27be-4576-a46e-23230b6d9c46 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for maintaining distance from the lead vehicle while going straight."""
    comp = {
        "perceived_lead_vehicle": 0.0,
        "committed_to_keep_distance": 0.0,
        "maintained_speed_and_trajectory": 0.0
    }

    # Check perceptual claims
    perceived_lead_vehicle = any(pc.entity == 'lead_vehicle' and pc.state == 'ahead' for pc in claims.perceptual)
    if perceived_lead_vehicle:
        comp["perceived_lead_vehicle"] = 0.10

    # Check commitment claims
    committed_to_keep_distance = any(cc.maneuver == 'keep_distance' and cc.speed_profile == 'maintain' for cc in claims.commitments)
    if committed_to_keep_distance:
        comp["committed_to_keep_distance"] = 0.10

    # Check combined trajectory execution
    # Speed maintenance: within ±1.5 m/s of the GT range and timing of speed drop
    speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
    if len(speed_window) > 0:
        min_speed = min(speed_window)
        min_speed_time = traj.dt_s * np.argmin(speed_window)
        if 18.3 <= traj.initial_speed_mps <= 21.3 and 18.3 <= traj.final_speed_mps <= 22.5:
            if min_speed_time > 3.0 and min_speed < traj.initial_speed_mps:
                if perceived_lead_vehicle and committed_to_keep_distance:
                    comp["maintained_speed_and_trajectory"] = 0.80

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
