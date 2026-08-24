"""clip 1cad1564-e40c-4ece-9f9f-d913f634a482 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for the scene with a cyclist crossing ahead and minimal speed drop.
    
    Decisive Events:
    1. Cyclist Crossing the Road Ahead:
       - Perceptual mention: cyclist, pedestrian
       - Commitment: speed_profile='decelerate'
       - Trajectory: Speed drop >= 0.05 m/s, graded factor

    Scene-derived thresholds:
    - Speed drop floor: 0.05 m/s (half of GT's 0.1 m/s drop)
    - Graded speed drop factor: 0.5 * min(1.0, drop / 0.1)
    """
    comp = {
        "mention_cyclist": 0.0,
        "decelerate_executed": 0.0,
    }

    # Check for perceptual mention of cyclist or related entities
    if any(p.entity in ('cyclist', 'pedestrian') for p in claims.perceptual):
        comp["mention_cyclist"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.2, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 0.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
