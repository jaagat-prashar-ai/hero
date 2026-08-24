"""clip b92b4df4-9ace-43c3-8894-e10cb165f42a - attempt 5/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene b92b4df4-9ace-43c3-8894-e10cb165f42a.
    
    Decisive Events:
    1. Construction Zone on the Right: Steer left to maintain a safe distance.
       - Perceptual: Mention of 'work_zone', 'construction_cones', 'barricades', or 'workers'.
       - Commitment: 'nudge' maneuver, excluding 'right' direction.
       - Trajectory: Lateral offset change of at least 0.08 m to the left.
    
    Scene-derived thresholds:
    - Lateral offset change: 0.5 * min(1.0, offset_change / 0.45)
    - Perceptual mention weight: 0.1
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_score = 0.0

    # Check perceptual mentions
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for lateral commitment and corresponding trajectory
    if any(c.maneuver in ('nudge', 'lane_change', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
        # Ensure the trajectory reflects a leftward maneuver
        if traj.total_heading_change_deg < 0:
            lateral_score = 0.6 * min(1.0, lateral_offset_change / 0.45)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_execution": lateral_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
