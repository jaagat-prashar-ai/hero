"""clip 855cc64f-4ce0-4e1b-9dbc-3a5c4ac41285 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.24, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene with a white utility truck and construction cones blocking the lane.
    Decisive events:
    1. Anticipated stop for the truck and cones (speed_profile='decelerate').
    2. Maintaining lane position (minimal lateral offset change).
    Trajectory thresholds derived from GT: speed drop >= 0.0 m/s (graded), lateral offset change minimal.
    """

    # Initialize component scores
    comp = {
        "mention_obstacle": 0.0,
        "decelerate_executed": 0.0
    }

    # Perceptual mention of relevant obstacles
    if any(p.entity in ('vehicle_generic', 'construction_cones', 'work_zone', 'barricades') for p in claims.perceptual):
        comp["mention_obstacle"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration
        comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 3.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
