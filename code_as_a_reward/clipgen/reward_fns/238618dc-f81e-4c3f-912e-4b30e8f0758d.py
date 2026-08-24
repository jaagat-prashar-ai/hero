"""clip 238618dc-f81e-4c3f-912e-4b30e8f0758d - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with steering left through a construction zone.
    
    Decisive events:
    1. Steering left through the construction zone.
       - Perceptual mention: construction-related entities.
       - Commitment: Lateral maneuver (lane_change, nudge, turn) excluding right.
       - Trajectory: Leftward lateral offset change of at least -0.15 m and heading change of at least -0.35 degrees, occurring towards the end of the window.
    """
    # Initialize component scores
    perceptual_construction = 0.0
    lateral_maneuver = 0.0

    # Check for perceptual mentions of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1  # Small additive weight for mention

    # Check for lateral maneuver commitment excluding right
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate graded lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        graded_lateral_offset = 0.45 * min(1.0, abs(lateral_offset_change) / 0.33)
        
        # Calculate graded heading change
        heading_change = traj.total_heading_change_deg
        graded_heading_change = 0.45 * min(1.0, abs(heading_change) / 0.7)
        
        # Ensure the maneuver occurs towards the end of the window
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        if min_speed_time >= 3.2:  # Ensure the maneuver happens in the latter half of the window
            lateral_maneuver = graded_lateral_offset + graded_heading_change

    return {
        "perceptual_construction": perceptual_construction,
        "lateral_maneuver": lateral_maneuver
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
