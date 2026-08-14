"""clip 5c2f2381-0e70-429d-a17c-8319e2156cd7 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on decisive events:
    1. Yield to pedestrian: Expect a 'decelerate' commitment and a speed drop of at least 1.0 m/s by t=6.3s.
    Perceptual mentions are scored separately with small weight.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_yield": 0.0,
    }

    # Perceptual mention of pedestrian
    if any(p.entity in ('pedestrian', 'cyclist') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment to yield (decelerate) and corresponding trajectory check
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Check the timing of the speed drop
        min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        min_speed_time = min_speed_idx * traj.dt_s

        # Graded factor for speed drop with timing consideration
        if min_speed_time >= 6.0:  # Expecting the yield to occur later in the window
            comp["commitment_yield"] = 0.9 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
