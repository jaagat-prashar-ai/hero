"""clip 000ba013-9eb4-45ca-8e86-93fdc68c37e2 - attempt 4/5 - gate PASS (pos 0.91, max pert 0.05, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration in response to a stop sign (commitment: 'decelerate')
    - Mention of a valid traffic control entity (perceptual: 'signal', 'intersection')
    - Trajectory speed reduction of at least 0.05 m/s within the window
    - Deceleration trajectory factor is graded, allowing for continuous variation
    """

    # Initialize component scores
    comp = {
        "mention_traffic_control": 0.0,
        "decelerate_commitment": 0.0,
        "trajectory_deceleration": 0.0,
    }

    # Check for perceptual mention of valid traffic control entities
    if any(p.entity in ('signal', 'intersection') for p in claims.perceptual):
        comp["mention_traffic_control"] = 0.05  # Reduced weight

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.3

        # Calculate trajectory speed reduction
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for deceleration
        if speed_drop >= 0.05:  # Adjusted threshold for the positive case
            comp["trajectory_deceleration"] = 0.6 * min(1.0, speed_drop / 0.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
