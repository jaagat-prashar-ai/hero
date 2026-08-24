"""clip ef491309-1c87-445c-800f-a6759e06dbb3 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 1)"""
def components(claims, traj):
    """Components for the scene where the ego vehicle stops behind a lead vehicle at a traffic light.
    Decisive events: stop behind lead vehicle, influenced by traffic light.
    Scene-derived thresholds: speed drop >= 0.05 m/s by t=0.5 s.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'signal') for p in claims.perceptual):
        perceptual_score = 0.05  # Mention-only credit

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory factor for deceleration
        trajectory_factor = min(1.0, speed_drop / 0.05)

        # Combine commitment and trajectory into a single score
        commitment_score = 0.65 * trajectory_factor

    # Return component contributions
    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
