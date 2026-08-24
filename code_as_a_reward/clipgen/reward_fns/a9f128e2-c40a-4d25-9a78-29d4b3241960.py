"""clip a9f128e2-c40a-4d25-9a78-29d4b3241960 - attempt 3/5 - gate PASS (pos 0.90, max pert 0.12, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Steering Right: Lateral maneuver to the right following delineators.
    - Speed Reduction: Deceleration in response to traffic or road conditions.
    - Perceptual Mentions: Recognition of relevant entities like delineators.
    Scene-derived thresholds: 
    - Lateral offset change >= 0.8 m (rightward)
    - Heading change >= 2.0 degrees (rightward)
    - Speed drop >= 1.55 m/s
    """
    # Initialize component scores
    components = {
        "steering_right": 0.0,
        "speed_reduction": 0.0,
        "mention_delineators": 0.0
    }

    # Check for perceptual mentions of delineators
    if any(p.entity in ('construction_cones', 'barricades', 'work_zone') for p in claims.perceptual):
        components["mention_delineators"] = 0.1

    # Check for lateral maneuver commitment to the right
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = abs(traj.final_lateral_offset_m) - abs(traj.lateral_offset_m[0])
        # Calculate heading change
        heading_change = abs(traj.total_heading_change_deg)
        # Graded lateral factor
        lateral_factor = 0.5 * min(1.0, lateral_offset_change / 6.37) + 0.3 * min(1.0, heading_change / 9.0)
        components["steering_right"] = lateral_factor

    # Check for speed reduction commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded speed reduction factor
        speed_reduction_factor = 0.4 * min(1.0, speed_drop / 3.1)
        components["speed_reduction"] = speed_reduction_factor

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
