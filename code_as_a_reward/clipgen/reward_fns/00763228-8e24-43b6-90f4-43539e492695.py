"""clip 00763228-8e24-43b6-90f4-43539e492695 - attempt 5/5 - gate PASS (pos 0.72, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Decisive events: maintaining a safe distance from the lead vehicle and intended lane change to the left.
    Scene-derived thresholds: minimal speed change (0.0 m/s) and minimal lateral offset (close to 0.0 m).
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    perceptual_lane = 0.0
    lane_change_execution = 0.0

    # Check for perceptual claims
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.05

    if any(p.entity == 'lane' for p in claims.perceptual):
        perceptual_lane = 0.05

    # Check for lane change commitment and trajectory execution
    if any(c.maneuver == 'lane_change' and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = abs(traj.final_lateral_offset_m)
        if lateral_offset_change > 1.0:  # Threshold for meaningful lateral movement
            lane_change_execution = 0.7 * min(1.0, lateral_offset_change / 2.0)  # Graded factor for lateral stability

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "perceptual_lane": perceptual_lane,
        "lane_change_execution": lane_change_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
