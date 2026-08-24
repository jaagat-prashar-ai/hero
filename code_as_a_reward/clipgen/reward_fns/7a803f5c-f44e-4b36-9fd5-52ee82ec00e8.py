"""clip 7a803f5c-f44e-4b36-9fd5-52ee82ec00e8 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration commitment and execution: checks for a deceleration claim and a graded speed drop.
    - Perceptual mention of pedestrian or related entities.
    - The scene involves gentle deceleration to maintain a safe distance from a pedestrian.
    - Trajectory thresholds are set to half the GT's magnitude for graded scoring.
    """
    # Initialize component scores
    deceleration_commitment_score = 0.0
    perceptual_mention_score = 0.0

    # Check for perceptual mention of pedestrian-related entities
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_mention_score = 0.1  # Small additive weight for mention

    # Check for deceleration commitment with correct speed change direction
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the speed decreases over time
        if traj.initial_speed_mps > traj.final_speed_mps:
            # Graded factor for deceleration execution
            deceleration_commitment_score = 0.6 * min(1.0, speed_drop / 0.7)  # Half of observed drop is 0.7 m/s

    return {
        "deceleration_commitment": deceleration_commitment_score,
        "perceptual_mention": perceptual_mention_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
