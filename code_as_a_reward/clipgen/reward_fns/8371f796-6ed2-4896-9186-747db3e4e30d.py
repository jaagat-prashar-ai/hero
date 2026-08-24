"""clip 8371f796-6ed2-4896-9186-747db3e4e30d - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene 8371f796-6ed2-4896-9186-747db3e4e30d:
    1. Steering left to follow road curvature while maintaining a safe distance from delineators.
       - Commitment: Lateral maneuver ('nudge', 'lane_change', 'turn', 'merge', 'enter', 'exit') excluding 'right'
       - Trajectory: Heading change >= -1.5 degrees, Lateral offset >= -0.5 m
    2. Maintaining speed.
       - Commitment: Speed profile 'maintain' or 'accelerate'
       - Trajectory: Speed drop <= 0.5 m/s
    """
    comp = {
        "lateral_maneuver": 0.0,
        "maintain_speed": 0.0,
    }

    # Lateral maneuver commitment and trajectory
    if any(c.maneuver in ('lane_change', 'nudge', 'turn', 'merge', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        heading_change = traj.total_heading_change_deg
        lateral_offset = traj.final_lateral_offset_m
        heading_factor = min(1.0, abs(heading_change) / 1.5)  # Graded factor for heading change
        lateral_factor = min(1.0, abs(lateral_offset) / 0.5)  # Graded factor for lateral offset
        comp["lateral_maneuver"] = 0.7 * min(heading_factor, lateral_factor)

    # Speed maintenance commitment and trajectory
    if any(c.speed_profile in ('maintain', 'accelerate') for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        speed_factor = 0.3 * min(1.0, max(0.0, (0.5 - speed_drop) / 0.5))  # Graded factor for speed maintenance
        comp["maintain_speed"] = speed_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
