"""clip 23504eb0-d082-4c71-8832-51b25add5dc8 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with road closure ahead:
    - Deceleration commitment with graded trajectory factor.
    - Perceptual mention of road-related entities.
    Trajectory thresholds: speed drop >= 0.05 m/s, graded.
    """
    comp = {
        "perceptual_mention": 0.0,
        "deceleration_commitment": 0.0,
    }

    # Perceptual mention of road-related entities
    if any(p.entity in ('barricades', 'construction_cones', 'work_zone', 'workers') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Deceleration commitment and trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, floor at 0.05 m/s
        comp["deceleration_commitment"] = 0.6 * min(1.0, speed_drop / 0.05)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
