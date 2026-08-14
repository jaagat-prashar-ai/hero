"""clip fc0f73fb-572e-4a4d-92ff-3a22ea203e10 - attempt 4/5 - gate PASS (pos 0.73, max pert 0.00, real rollout argmax 10)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of navigating through a construction zone.
    - Perceptual: Mentions of construction-related entities.
    - Commitment: Maintaining a straight path through the zone.
    - Trajectory: Steady or slightly increasing speed, minimal lateral offset, and minimal heading change.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_and_trajectory_score = 0.0

    # Perceptual component: Mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Commitment component: Check for a relevant commitment claim
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Trajectory component: Steady or slightly increasing speed
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase > 0:
            speed_factor = 0.5 * min(1.0, speed_increase / 2.0)  # Floor at half the GT increase
            commitment_and_trajectory_score += speed_factor

            # Lateral offset: Minimal deviation
            max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
            lateral_factor = 0.2 * (1.0 - min(1.0, max_lateral_offset / 0.54))  # Floor at half the GT max offset
            commitment_and_trajectory_score += lateral_factor

            # Heading change: Minimal change
            heading_change = abs(traj.total_heading_change_deg)
            heading_factor = 0.2 * (1.0 - min(1.0, heading_change / 0.4))  # Floor at half the GT heading change
            commitment_and_trajectory_score += heading_factor

    return {
        "perceptual": perceptual_score,
        "commitment_and_trajectory": commitment_and_trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
