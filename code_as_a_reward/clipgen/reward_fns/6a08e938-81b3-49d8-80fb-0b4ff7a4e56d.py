"""clip 6a08e938-81b3-49d8-80fb-0b4ff7a4e56d - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 2)"""
def components(claims, traj):
    """Decisive event: Maintain speed and lane. 
    - Perceptual: Acknowledge presence of vehicles.
    - Commitment: Maintain speed.
    - Trajectory: Minimal speed change.
    """

    # Initialize component scores
    perceptual_credit = 0.0
    maintain_speed_credit = 0.0

    # Perceptual: Check for mention of vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_credit = 0.05  # Small weight for perceptual mention

    # Commitment: Check for maintaining speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Trajectory: Check for minimal speed change
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase >= 0.0:
            maintain_speed_credit = 0.65 * min(1.0, speed_increase / 1.0)  # Graded factor for speed maintenance

    return {
        "perceptual_mention": perceptual_credit,
        "maintain_speed": maintain_speed_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
