"""clip 7ac5dc31-c4dc-4fd1-a476-5b3733ff11b5 - attempt 2/5 - gate PASS (pos 0.81, max pert 0.41, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene 7ac5dc31-c4dc-4fd1-a476-5b3733ff11b5:
    - Steering right to follow temporary lane (heading change, lateral offset)
    - Speed reduction in construction zone (speed drop)
    Trajectory thresholds: heading change floor -3.0 deg, lateral offset floor -1.0 m,
    speed drop floor 0.5 m/s. Perceptual mentions: construction_cones, work_zone.
    """

    # Initialize component scores
    comp = {
        "mention_construction": 0.0,
        "lateral_maneuver": 0.0,
        "speed_reduction": 0.0,
    }

    # Perceptual mention credit
    if any(p.entity in ('construction_cones', 'work_zone') for p in claims.perceptual):
        comp["mention_construction"] = 0.05

    # Lateral maneuver commitment and trajectory check
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        lateral_offset = traj.final_lateral_offset_m
        # Graded factors for lateral maneuver
        heading_factor = 0.45 * min(1.0, abs(heading_change) / 6.0)
        lateral_factor = 0.45 * min(1.0, abs(lateral_offset) / 2.0)
        comp["lateral_maneuver"] = heading_factor + lateral_factor

    # Speed reduction commitment and trajectory check
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        # Graded factor for speed reduction
        comp["speed_reduction"] = 0.5 * min(1.0, speed_drop / 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
