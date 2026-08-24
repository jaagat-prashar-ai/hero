"""clip 75b8dd37-0f11-44af-bf3c-79f0b41c4052 - attempt 1/5 - gate PASS (pos 0.77, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scoring a rollout's faithfulness to the scene:
    - Deceleration: Expect a speed drop of at least 3.8 m/s, with graded credit for greater drops.
    - Pedestrian Mention: Credit for mentioning 'pedestrian' or 'crosswalk'.
    - Yield Commitment: Credit for a commitment to decelerate (stop/yield/wait/decelerate).
    """
    comp = {
        "deceleration": 0.0,
        "pedestrian_mention": 0.0,
        "yield_commitment": 0.0,
    }

    # Check for perceptual mention of pedestrians or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["pedestrian_mention"] = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded credit for speed drop, with a floor at 3.8 m/s
        if speed_drop >= 3.8:
            comp["deceleration"] = 0.5 * min(1.0, speed_drop / 7.6)
            comp["yield_commitment"] = 0.4

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
