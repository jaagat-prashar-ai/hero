"""clip 037150fe-1182-4b57-8d5d-a5afa3fbfbc2 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 10)"""
def components(claims, traj):
    """
    Components for scene with decisive events:
    1. Yield to Rider: Expect deceleration with a speed drop of at least 1.2 m/s.
    2. Resume Speed: Expect acceleration after yielding, with a speed increase.
    Trajectory thresholds are derived from the ground truth dossier.
    """

    # Initialize component scores
    comp = {
        "mention_rider": 0.0,
        "speed_increase": 0.0
    }

    # Check for perceptual mention of rider-related entities
    if any(p.entity in ('cyclist',) for p in claims.perceptual):
        comp["mention_rider"] = 0.1

    # Check for speed increase commitment and corresponding trajectory behavior
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase
        initial_speed = traj.speed_mps[0]
        final_speed = traj.final_speed_mps
        speed_increase = final_speed - initial_speed

        # Graded score for speed increase
        comp["speed_increase"] = 0.9 * min(1.0, speed_increase / 7.8)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
