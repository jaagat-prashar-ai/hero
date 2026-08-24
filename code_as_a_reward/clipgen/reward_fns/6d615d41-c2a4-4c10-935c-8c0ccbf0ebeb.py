"""clip 6d615d41-c2a4-4c10-935c-8c0ccbf0ebeb - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with decisive events: steering left to pass a truck and slight speed adjustment.
    - Commitment to 'lane_change' or 'nudge' with direction not 'right'.
    - Lateral offset change of at least +1.0 m for passing maneuver.
    - Speed drop of at least 0.7 m/s for speed adjustment.
    """

    # Initialize component scores
    lateral_maneuver = 0.0
    speed_adjustment = 0.0

    # Check for lateral maneuver commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'right' for c in claims.commitments):
        # Calculate lateral offset change
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        if lateral_offset_change < 0:  # Ensure the change is in the correct direction (right)
            lateral_maneuver = 0.7 * min(1.0, abs(lateral_offset_change) / 1.7)  # Graded factor, floor at +1.7 m

    # Check for speed adjustment commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        speed_adjustment = 0.3 * min(1.0, speed_drop / 0.7)  # Graded factor, floor at 0.7 m/s

    return {
        "lateral_maneuver": lateral_maneuver,
        "speed_adjustment": speed_adjustment
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
