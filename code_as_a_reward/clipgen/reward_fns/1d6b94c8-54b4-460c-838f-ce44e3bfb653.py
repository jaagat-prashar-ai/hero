"""clip 1d6b94c8-54b4-460c-838f-ce44e3bfb653 - attempt 5/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scene 1d6b94c8-54b4-460c-838f-ce44e3bfb653:
    - Maintain speed with lead vehicle: perceptual mention of lead vehicle or vehicle_generic,
      commitment to maintain speed, trajectory maintaining speed with minimal drop.
    Thresholds derived from GT: minimal speed drop (<= 0.5 m/s).
    """

    # Initialize component scores
    comp = {
        "mention_lead_vehicle": 0.0,
        "maintain_speed": 0.0,
    }

    # Perceptual mention of lead vehicle or vehicle_generic
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_lead_vehicle"] = 0.1

    # Commitment to maintain speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Trajectory maintaining speed with minimal drop
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase >= 0.5:
            comp["maintain_speed"] = 0.7 * min(1.0, speed_increase / 1.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
