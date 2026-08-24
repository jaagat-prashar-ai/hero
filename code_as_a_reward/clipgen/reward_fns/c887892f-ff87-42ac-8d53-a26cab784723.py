"""clip c887892f-ff87-42ac-8d53-a26cab784723 - attempt 3/5 - gate PASS (pos 0.75, max pert 0.05, real rollout argmax 11)"""
def components(claims, traj):
    """Components for the scene where the ego vehicle must stop and yield to a pedestrian.
    
    Decisive Event:
    1. Stop and yield to the pedestrian (Track 10).
       - Perceptual mention of 'pedestrian'.
       - Commitment to 'decelerate' (stop/yield/wait/decelerate).
       - Trajectory should maintain a low speed, reflecting a stop/yield action.
    
    Trajectory thresholds:
    - Speed should remain below 0.5 m/s for full credit, graded from 0.5 m/s.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_decelerate = 0.0

    # Check for perceptual mention of pedestrian
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        perceptual_pedestrian = 0.05  # Reduced weight for mention-only credit

    # Check for commitment to decelerate (stop/yield/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop factor
        min_speed = min(traj.speed_mps)
        speed_factor = 0.7 * min(1.0, (0.5 - min_speed) / 0.5)
        commitment_decelerate = speed_factor

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_decelerate": commitment_decelerate,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
