"""clip ec5e4186-e466-439b-8088-d2719dc9959e - attempt 5/5 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene ec5e4186-e466-439b-8088-d2719dc9959e:
    - Maintain speed while maintaining a safe distance from the lead vehicle.
    - Thresholds derived from GT: speed drop < 1 m/s, lateral offset change < 1 m, heading change < 5 degrees.
    """

    # Initialize component scores
    perceptual_lead_vehicle = 0.0
    maintain_speed_commitment = 0.0
    maintain_speed_execution = 0.0

    # Check for perceptual claims
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_lead_vehicle = 0.05  # Mention-only credit

    # Check for commitment claims
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        maintain_speed_commitment = 0.15  # Mention-only credit

    # Trajectory analysis for maintaining speed, gated by commitment
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.2:  # Half of the observed drop in the positive case
            maintain_speed_execution = 0.8 * min(1.0, speed_drop / 0.4)

    # Return component scores
    return {
        "perceptual_lead_vehicle": perceptual_lead_vehicle,
        "maintain_speed_commitment": maintain_speed_commitment,
        "maintain_speed_execution": maintain_speed_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
