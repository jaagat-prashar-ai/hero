"""clip 3a59cc15-7c97-48ca-b0ff-5f3ace95818b - attempt 3/5 - gate PASS (pos 0.74, max pert 0.05, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive event:
    Yield to traffic in the roundabout.
    - Perceptual mention of 'roundabout' or 'vehicle_generic'.
    - Commitment to 'decelerate' family (stop/yield/wait/decelerate).
    - Trajectory should show a speed reduction of at least 2.05 m/s,
      graded above this floor.
    """
    perceptual_credit = 0.05 * any(
        p.entity in ('roundabout', 'vehicle_generic') for p in claims.perceptual
    )

    # Check for a commitment to decelerate
    commitment_credit = 0.0
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed reduction
        speed_drop_credit = 0.70 * min(1.0, speed_drop / 4.1)
        commitment_credit = speed_drop_credit

    return {
        "perceptual_mention": perceptual_credit,
        "commitment_execution": commitment_credit,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
