"""clip 8ea7fb77-5502-490b-ba1b-da6bdab98d8c - attempt 2/5 - gate PASS (pos 0.85, max pert 0.19, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and lead vehicle:
    - Decelerate to yield to pedestrian (speed drop >= 0.15 m/s)
    - Maintain safe distance from lead vehicle (speed drop >= 0.15 m/s)
    - Perceptual mentions: pedestrian, crosswalk, lead vehicle
    """
    # Initialize components
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "decelerate_for_lead_vehicle": 0.0,
        "mention_pedestrian": 0.05
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Check for deceleration commitment for pedestrian
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.3, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        if speed_drop >= 0.15:
            comp["decelerate_for_pedestrian"] = 0.4 * min(1.0, speed_drop / 4.7)

    # Check for deceleration commitment for lead vehicle
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.1, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        if speed_drop >= 0.15:
            comp["decelerate_for_lead_vehicle"] = 0.4 * min(1.0, speed_drop / 4.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
