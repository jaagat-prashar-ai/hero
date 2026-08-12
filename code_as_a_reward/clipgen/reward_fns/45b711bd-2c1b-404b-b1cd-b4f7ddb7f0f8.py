"""clip 45b711bd-2c1b-404b-b1cd-b4f7ddb7f0f8 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 3)"""
def components(claims, traj):
    # Initialize component scores
    comp = {
        "perceive_right_lane_vehicles": 0.0,
        "commit_safe_distance": 0.0,
        "execute_speed_reduction_and_stop": 0.0,
        "execute_lateral_control": 0.0
    }

    # Check for perceptual claims about right lane vehicles
    perceived_right_lane_vehicles = any(
        pc.entity in ["lead_vehicle", "vehicle_generic"] for pc in claims.perceptual
    )
    if perceived_right_lane_vehicles:
        comp["perceive_right_lane_vehicles"] = 0.1

    # Check for commitment to maintain safe distance
    commit_safe_distance = any(
        cc.maneuver == "keep_distance" and cc.speed_profile == "maintain"
        for cc in claims.commitments
    )
    if commit_safe_distance:
        comp["commit_safe_distance"] = 0.1

    # Check for speed reduction and stop execution with corresponding claims
    if traj.n_waypoints > 0:
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        min_speed = traj.min_speed_mps
        speed_reduction = initial_speed - final_speed
        stop_event = traj.stop_event

        # Ensure speed reduction and stop occur with claims
        if 0.7 <= speed_reduction <= 1.0 and min_speed <= 0.3 and stop_event:
            if perceived_right_lane_vehicles and commit_safe_distance:
                comp["execute_speed_reduction_and_stop"] = 0.5

    # Check for lateral control execution with corresponding claims
    final_lateral_offset = traj.final_lateral_offset_m
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    if abs(final_lateral_offset) <= 0.3 and max_lateral_offset <= 0.3:
        if perceived_right_lane_vehicles:
            comp["execute_lateral_control"] = 0.3

    return comp

def reward(claims, traj):
    """Decisive events: proximity to right lane vehicles, speed reduction, and stop.
    Scene-derived thresholds: speed reduction 0.7-1.0 m/s, min speed <= 0.3 m/s,
    lateral offset <= 0.3 m."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
