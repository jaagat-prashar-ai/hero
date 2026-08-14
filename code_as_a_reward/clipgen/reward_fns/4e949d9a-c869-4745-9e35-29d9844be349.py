"""clip 4e949d9a-c869-4745-9e35-29d9844be349 - attempt 1/5 - gate PASS (pos 0.96, max pert 0.49, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene 4e949d9a-c869-4745-9e35-29d9844be349:
    - Deceleration in response to a lead vehicle or traffic light, with a speed drop threshold of 4.4 m/s.
    - Maintaining lane position with a lateral offset within |0.72 m|.
    """
    comp = {
        "mention_lead_vehicle_or_signal": 0.0,
        "decelerate_executed": 0.0,
        "maintain_lane_position": 0.0
    }

    # Perceptual mention of lead vehicle or traffic light
    if any(p.entity in ('lead_vehicle', 'signal') for p in claims.perceptual):
        comp["mention_lead_vehicle_or_signal"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for deceleration execution
        comp["decelerate_executed"] = 0.5 * min(1.0, speed_drop / 8.8)

    # Maintaining lane position
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    if max_lateral_offset <= 0.72:
        # Graded factor for maintaining lane position
        comp["maintain_lane_position"] = 0.4 * min(1.0, (0.72 - max_lateral_offset) / 0.72)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
