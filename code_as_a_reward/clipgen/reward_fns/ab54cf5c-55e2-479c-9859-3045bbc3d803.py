"""clip ab54cf5c-55e2-479c-9859-3045bbc3d803 - attempt 1/5 - gate PASS (pos 0.90, max pert 0.50, real rollout argmax 9)"""
def components(claims, traj):
    """Components for reward calculation based on decisive events:
    1. Intersection Navigation: Gentle acceleration through the intersection.
       - Perceptual mention: {'construction_cones', 'barricades'}
       - Commitment: {'accelerate'}
       - Trajectory: Speed increase of at least 3.0 m/s over the window.
    2. Proximity to Construction Barriers and Traffic Cones: Navigating safely.
       - Perceptual mention: {'construction_cones', 'barricades'}
       - Commitment: {'turn'}
       - Trajectory: Heading change of at least -46.2 degrees.
    """

    # Initialize component scores
    perceptual_construction = 0.0
    accelerate_execution = 0.0
    turn_execution = 0.0

    # Check perceptual claims for construction-related entities
    if any(p.entity in ('construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_construction = 0.1

    # Check for acceleration commitment and corresponding trajectory execution
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        accelerate_execution = 0.5 * min(1.0, speed_increase / 6.0)

    # Check for turn commitment and corresponding trajectory execution
    if any(c.maneuver in ('turn', 'lane_change', 'nudge', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        turn_execution = 0.4 * min(1.0, abs(heading_change) / 46.2)

    return {
        "perceptual_construction": perceptual_construction,
        "accelerate_execution": accelerate_execution,
        "turn_execution": turn_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
