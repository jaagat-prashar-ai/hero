"""clip a4276b33-b02f-4ae9-9650-de1c47f0c423 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.20, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for reward function based on the scene:
    - Decisive Event 1: Pedestrian Crossing
      - Perceptual mention of 'pedestrian'
      - Commitment to 'decelerate'
      - Trajectory should show a slight deceleration (graded)
    - Decisive Event 2: Protruding Object
      - No specific trajectory or commitment required within the window
    """
    # Initialize component scores
    comp = {
        "mention_pedestrian": 0.0,
        "decelerate_commitment": 0.0,
        "decelerate_execution": 0.0,
    }

    # Check for perceptual mention of 'pedestrian'
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory execution: graded deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop > 0.5:  # Floor at half the scene's magnitude
            comp["decelerate_execution"] = 0.5 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
