"""clip 51bc1dd7-1188-4412-90ea-2939f90cb377 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with traffic barrels: maintain straight path and speed."""
    comp = {
        "maintain_speed": 0.0,
        "maintain_lane": 0.0,
    }

    # Maintain speed: check for speed_profile 'maintain' or 'decelerate'
    if any(c.speed_profile in ('maintain', 'decelerate') for c in claims.commitments):
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_maintenance = final_speed - initial_speed
        comp["maintain_speed"] = 0.7 * min(1.0, speed_maintenance / 6.4)

    # Maintain lane: minimal lateral deviation
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    if any(c.maneuver in ('keep_lane', 'nudge') for c in claims.commitments):
        comp["maintain_lane"] = 0.3 * min(1.0, (0.10 - max_lateral_offset) / 0.10)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
