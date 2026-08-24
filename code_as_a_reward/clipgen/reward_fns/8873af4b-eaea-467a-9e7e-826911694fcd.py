"""clip 8873af4b-eaea-467a-9e7e-826911694fcd - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 8873af4b-eaea-467a-9e7e-826911694fcd:
    - Curved Road Navigation: Expect lateral maneuver (rightward) with heading change.
    - Gentle Deceleration for Construction Zone: Expect deceleration intent with slight speed reduction.
    - Trajectory thresholds: heading change >= -20 degrees, speed drop >= 1 m/s.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0
    commitment_lateral = 0.0
    trajectory_lateral = 0.0

    # Perceptual check for construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Commitment and trajectory check for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.0:
            trajectory_decelerate = 0.4 * min(1.0, speed_drop / 2.0)  # Graded factor for speed drop >= 1 m/s
            commitment_decelerate = 0.2

    # Commitment and trajectory check for lateral maneuver (rightward)
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        if heading_change <= -20.0:
            trajectory_lateral = 0.3 * min(1.0, abs(heading_change) / 20.0)  # Graded factor for heading change >= -20 degrees
            commitment_lateral = 0.2

    # Return component scores
    return {
        "perceptual_construction": perceptual_construction,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate,
        "commitment_lateral": commitment_lateral,
        "trajectory_lateral": trajectory_lateral
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
