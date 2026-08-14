"""clip 299ca01c-772f-4b76-b14b-63406d52930b - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Deceleration and Stop: Expect a commitment to 'decelerate'.
       Trajectory should show a speed drop of at least 0.9 m/s by t=3.9 s.
    2. Maintaining Lateral Position: Expect no significant lateral movement outside -0.30 m to 0.30 m.
    """

    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0,
        "maintain_lateral_position": 0.0
    }

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 3.9))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 1.8)

    # Check for maintaining lateral position
    lateral_offsets = window(traj.lateral_offset_m, traj.dt_s, 0, 6.4)
    if lateral_offsets.size > 0:
        max_lateral_offset = max(abs(lateral_offsets))
        if max_lateral_offset <= 0.30:
            comp["maintain_lateral_position"] = 0.1

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
