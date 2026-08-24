"""clip 039de86d-1797-4c5e-ac45-d1825be666d0 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 039de86d-1797-4c5e-ac45-d1825be666d0:
    Maintaining speed with lead motorcycles. Thresholds derived from
    GT: speed increase from 4.5 to 6.7 m/s, minimal lateral offset.
    """
    # Initialize component scores
    perceptual_credit = 0.0
    maintain_speed_credit = 0.0

    # Perceptual entity check: any mention of relevant entities
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        perceptual_credit = 0.1  # Small weight for perceptual mention

    # Commitment check: maintaining speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Trajectory check: speed increase
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        maintain_speed_credit = 0.6 * min(1.0, speed_increase / 1.1)  # Adjusted graded factor

    return {
        "perceptual_mention": perceptual_credit,
        "maintain_speed_execution": maintain_speed_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
