"""clip 629f17bf-f5a4-4f43-bb27-982010264c7d - attempt 4/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration to yield right-of-way: Expect a 'decelerate' commitment and a speed drop of at least 0.75 m/s, with directional consideration.
    - Perceptual mention of nearby vehicles: Expect mention of 'vehicle_generic' or related entities.
    - Trajectory thresholds are derived from the expert's trajectory, with graded factors for speed drop.
    """

    # Initialize component scores
    deceleration_commitment_score = 0.0
    perceptual_mention_score = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop, with a floor at 0.75 m/s
        if traj.total_heading_change_deg > 0:  # Ensure correct directional change
            deceleration_commitment_score = 0.6 * min(1.0, speed_drop / 0.75)

    # Check for perceptual mention of nearby vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        perceptual_mention_score = 0.1

    return {
        "deceleration_commitment": deceleration_commitment_score,
        "perceptual_mention": perceptual_mention_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
