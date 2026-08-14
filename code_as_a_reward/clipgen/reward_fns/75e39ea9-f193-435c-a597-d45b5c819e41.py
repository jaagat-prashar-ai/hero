"""clip 75e39ea9-f193-435c-a597-d45b5c819e41 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring a rollout based on yielding to cyclists.
    
    Decisive event:
    1. Yield to cyclists (track 105 - Rider): Expect a 'decelerate' commitment and mention of 'cyclist'.
    
    Scene-derived thresholds:
    - Yield speed drop: floor at 0.15 m/s (half of GT's 0.3 m/s drop), with timing consideration.
    """

    # Initialize component scores
    scores = {
        "perceptual_cyclist": 0.0,
        "yield_executed": 0.0,
    }

    # Check for perceptual mention of cyclists
    if any(p.entity in ('cyclist',) for p in claims.perceptual):
        scores["perceptual_cyclist"] = 0.1

    # Check for yield commitment and corresponding trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop and timing
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))

        # Graded factor for yield execution with timing consideration
        if min_speed_time > 2.0:  # Ensure the drop occurs later in the window
            scores["yield_executed"] = 0.7 * min(1.0, speed_drop / 0.3)

    return scores

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
