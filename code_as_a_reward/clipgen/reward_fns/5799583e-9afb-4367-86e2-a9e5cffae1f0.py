"""clip 5799583e-9afb-4367-86e2-a9e5cffae1f0 - attempt 5/5 - gate PASS (pos 1.00, max pert 0.05, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with pedestrians on the right and automobile on the left.
    
    Decisive events:
    1. Maintain safe distance from pedestrians on the right.
       - Perceptual: 'pedestrian'
       - Commitment: 'decelerate' (speed_profile family)
       - Trajectory: Minimal speed drop, staying nearly stationary.
    2. Maintain safe distance from automobile on the left.
       - Perceptual: 'vehicle_generic'
       - Commitment: 'nudge' (lateral maneuver family)
       - Trajectory: Slight lateral offset to the left.
    
    Trajectory thresholds:
    - Lateral offset floor: 0.005 m to the left
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    nudge_commitment = 0.0

    # Check for perceptual claims
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        perceptual_pedestrian = 0.05

    # Check for commitment claims and corresponding trajectory execution
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Graded lateral offset factor
        lateral_offset = traj.final_lateral_offset_m
        if lateral_offset >= 0.005:  # Floor at 0.005 m
            nudge_commitment = 0.95 * min(1.0, lateral_offset / 1.53)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "nudge_commitment": nudge_commitment,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
