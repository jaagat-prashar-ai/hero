"""clip ad2254be-b493-46fc-b45d-e2d85a50154d - attempt 1/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene ad2254be-b493-46fc-b45d-e2d85a50154d:
    - Deceleration in response to protruding object and other vehicle on the right.
    - Perceptual mention of construction-related entities.
    - Graded speed reduction with a floor at half the GT's magnitude.
    """
    comp = {
        "mention_construction": 0.0,
        "decelerate_response": 0.0,
    }

    # Perceptual mention of construction-related entities
    if any(p.entity in ('construction_cones', 'work_zone', 'barricades', 'workers') for p in claims.perceptual):
        comp["mention_construction"] = 0.1

    # Deceleration response to obstacles
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded speed reduction factor
        comp["decelerate_response"] = 0.7 * min(1.0, speed_drop / 6.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
