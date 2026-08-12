"""clip d05dcfff-5aee-4402-85fa-2cc1cdbd2545 - attempt 3/3 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 4)"""
def components(claims, traj):
    # Initialize component scores
    components = {
        "committed_to_yield_with_execution": 0.0,
        "executed_deceleration_with_claim": 0.0,
        "maintained_lateral_stability": 0.0
    }

    # Check for commitment to yield with matching execution
    committed_to_yield = any(
        cc.maneuver == 'yield' and cc.speed_profile == 'decelerate'
        for cc in claims.commitments
    )
    if committed_to_yield and traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        min_speed = traj.min_speed_mps

        # Check if the speed drops significantly and at the correct time
        if initial_speed - final_speed >= 4.0 and min_speed <= 0.5 and np.argmin(speed_window) * traj.dt_s >= 6.0:
            components["committed_to_yield_with_execution"] = 0.5

    # Check for executed deceleration with matching claim
    saw_pedestrian = any(
        pc.entity == 'pedestrian' and pc.state == 'crossing'
        for pc in claims.perceptual
    )
    if saw_pedestrian and traj.n_waypoints > 0:
        if initial_speed - final_speed >= 4.0 and min_speed <= 0.5:
            components["executed_deceleration_with_claim"] = 0.3

    # Check for lateral stability
    if traj.n_waypoints > 0:
        lateral_window = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
        max_lateral_offset = np.max(np.abs(lateral_window))

        # Ensure lateral offset does not exceed a reasonable threshold
        if max_lateral_offset <= 30.0:
            components["maintained_lateral_stability"] = 0.2

    return components

def reward(claims, traj):
    """Reward function for scene with pedestrians crossing at a crosswalk.
    Decisive events include recognizing pedestrians, committing to yield,
    executing a deceleration, and maintaining lateral stability. Thresholds
    are based on the expert trajectory's speed drop and lateral offset."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
