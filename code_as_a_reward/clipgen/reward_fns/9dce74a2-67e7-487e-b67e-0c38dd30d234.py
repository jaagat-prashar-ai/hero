"""clip 9dce74a2-67e7-487e-b67e-0c38dd30d234 - attempt 1/5 - gate PASS (pos 0.90, max pert 0.20, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with deceleration due to construction zone.
    
    Decisive Event: Deceleration in response to construction zone.
    - Perceptual mention: work_zone, construction_cones, barricades, workers.
    - Commitment: Decelerate (speed_profile='decelerate').
    - Trajectory: Speed drop of at least 3.2 m/s, graded with higher scores for greater drops.
    """
    comp = {
        "perceptual_mention": 0.0,
        "decelerate_commitment": 0.0,
        "trajectory_deceleration": 0.0
    }

    # Perceptual mention component
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory deceleration component
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 3.2:
            # Graded factor for speed drop
            comp["trajectory_deceleration"] = 0.7 * min(1.0, speed_drop / 6.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
