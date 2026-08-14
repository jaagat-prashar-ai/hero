"""clip b2e77e81-71fc-4f41-9ee7-659f9bfc1ae3 - attempt 1/5 - gate PASS (pos 0.86, max pert 0.39, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Gentle deceleration in response to nearby vehicles.
       - Perceptual mention of 'vehicle_generic'.
       - Commitment to 'decelerate'.
       - Trajectory should show a speed drop of at least 1.6 m/s.
    2. Lateral stability.
       - Trajectory should maintain lateral offset within ±0.30 m.
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    deceleration_commitment = 0.0
    lateral_stability = 0.0

    # Check for perceptual mention of nearby vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop
        deceleration_commitment = 0.5 * min(1.0, speed_drop / 3.2)

    # Check for lateral stability
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    if max_lateral_offset <= 0.30:
        lateral_stability = 0.4 * (1.0 - min(1.0, max_lateral_offset / 0.30))

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "deceleration_commitment": deceleration_commitment,
        "lateral_stability": lateral_stability
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
