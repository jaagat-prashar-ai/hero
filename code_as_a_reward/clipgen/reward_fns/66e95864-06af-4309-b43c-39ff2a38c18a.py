"""clip 66e95864-06af-4309-b43c-39ff2a38c18a - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene with gentle acceleration and maintaining safe distance in a construction zone."""
    comp = {
        "perceptual_construction": 0.0,
        "commitment_accelerate": 0.0,
    }

    # Perceptual component: construction-related entities
    if any(p.entity in ('workers', 'construction_cones', 'work_zone') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Commitment component: gentle acceleration
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Graded trajectory factor for acceleration
        speed_gain = traj.final_speed_mps - traj.initial_speed_mps
        comp["commitment_accelerate"] = 0.9 * min(1.0, speed_gain / 4.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
