"""clip d68ecfb2-dce0-4ea9-b89a-c6a0ab1367c4 - attempt 2/5 - gate PASS (pos 0.85, max pert 0.10, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene: gentle acceleration through a construction zone.
    Decisive event: gentle acceleration while following the temporary lane.
    - Perceptual mention of construction-related entities.
    - Commitment to accelerate with matching trajectory execution.
    - Trajectory shows speed increase of at least 0.8 m/s.
    """
    perceptual_weight = 0.1
    commitment_weight = 0.6
    trajectory_weight = 0.3

    # Perceptual component: any mention of construction-related entities
    perceptual_mention = any(
        p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers')
        for p in claims.perceptual
    )

    # Commitment component: commitment to accelerate with matching trajectory execution
    commitment_to_accelerate = any(
        c.speed_profile == 'accelerate' for c in claims.commitments
    ) and (traj.final_speed_mps - traj.initial_speed_mps > 0.8)

    # Trajectory component: graded speed increase
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    graded_speed_increase = 0.5 * min(1.0, speed_increase / 1.6)

    # Components dictionary
    components = {
        "perceptual_mention": perceptual_weight if perceptual_mention else 0.0,
        "commitment_to_accelerate": commitment_weight if commitment_to_accelerate else 0.0,
        "trajectory_speed_increase": trajectory_weight * graded_speed_increase if commitment_to_accelerate else 0.0,
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
