"""clip 29569203-586b-49c2-832c-b993e0b14985 - attempt 1/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness in navigating a construction zone.
    Decisive event: Navigating through the construction zone on the right side of the road.
    - Perceptual: Mention of 'work_zone', 'construction_cones', 'barricades', or 'workers'.
    - Commitment: Maintain speed (speed_profile='maintain').
    - Trajectory: Minimal speed drop, graded with a floor at half the GT's drop.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Perceptual component: Check for mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Commitment component: Check for a commitment to maintain speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Trajectory component: Check for minimal speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor, floor at half the GT's drop (0.15 m/s)
        trajectory_score = 0.5 * min(1.0, speed_drop / 0.3)
        commitment_score = 0.4

    # Combine components
    return {
        "perceptual": perceptual_score,
        "commitment_and_trajectory": commitment_score + trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
