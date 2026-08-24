"""clip 3b9a3679-a29c-49a6-847f-111633203764 - attempt 2/5 - gate PASS (pos 0.77, max pert 0.30, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene 3b9a3679-a29c-49a6-847f-111633203764:
    - Deceleration to maintain a safe distance from a pedestrian.
    - Proximity to nearby vehicles.
    Trajectory thresholds: speed drop >= 3.0 m/s (graded), perceptual mentions
    of 'pedestrian' or 'vehicle_generic', commitment family 'decelerate'.
    """
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_vehicle": 0.0,
        "decelerate_commitment": 0.0,
        "speed_drop": 0.0,
    }

    # Perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory-based speed drop
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded speed drop factor
        if speed_drop >= 3.0:
            comp["speed_drop"] = 0.5 * min(1.0, speed_drop / 6.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
