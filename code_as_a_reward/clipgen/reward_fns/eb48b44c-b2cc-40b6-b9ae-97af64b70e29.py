"""clip eb48b44c-b2cc-40b6-b9ae-97af64b70e29 - attempt 1/5 - gate PASS (pos 0.97, max pert 0.56, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scene with lane change to the left and maintaining speed.
    Decisive events:
    1. Lane Change to the Left: Expect a 'lane_change' commitment with a left direction,
       and a lateral offset change of at least +3.9 m.
    2. Maintaining Speed: No specific commitment required, but speed should remain
       within 13.3 m/s of the initial speed.
    """
    comp = {
        "mention_vehicle": 0.0,
        "mention_lane": 0.0,
        "lane_change_executed": 0.0,
        "maintain_speed": 0.0,
    }

    # Perceptual mentions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        comp["mention_vehicle"] = 0.05

    if any(p.entity == 'lane' for p in claims.perceptual):
        comp["mention_lane"] = 0.05

    # Lane change commitment and execution
    if any(c.maneuver == 'lane_change' and c.direction != 'right' for c in claims.commitments):
        lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
        comp["lane_change_executed"] = 0.5 * min(1.0, lateral_offset_change / 7.83)

    # Maintaining speed
    speed_maintenance = abs(traj.initial_speed_mps - traj.final_speed_mps)
    comp["maintain_speed"] = 0.4 * min(1.0, (26.6 - speed_maintenance) / 26.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
