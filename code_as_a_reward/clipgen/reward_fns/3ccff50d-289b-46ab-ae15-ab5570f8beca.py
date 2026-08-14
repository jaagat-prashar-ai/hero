"""clip 3ccff50d-289b-46ab-ae15-ab5570f8beca - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scene with decisive events:
    1. Steering right through construction zone: expect a rightward maneuver
       with a heading change of at least -5 degrees and a lateral offset of
       at least -1.5 m. Perceptual mentions: construction-related entities.
    2. Speed adjustment: expect a speed drop of at least 1.1 m/s. Perceptual
       mentions: pedestrians, vehicles, or obstacles.
    """

    # Initialize component scores
    comp = {
        "rightward_maneuver": 0.0,
        "speed_deceleration": 0.0
    }

    # Rightward maneuver: lane_change/nudge/merge/turn/enter/exit with direction not left
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        lateral_offset = traj.final_lateral_offset_m
        # Graded factor for heading change
        heading_factor = 0.3 * min(1.0, abs(heading_change) / 10.0)
        # Graded factor for lateral offset
        lateral_factor = 0.3 * min(1.0, abs(lateral_offset) / 2.99)
        comp["rightward_maneuver"] = heading_factor + lateral_factor

    # Speed deceleration: decelerate family
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        # Graded factor for speed drop
        comp["speed_deceleration"] = 0.7 * min(1.0, speed_drop / 2.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
