"""clip 9bcf4979-b04b-4bf3-8ad6-6e35e3be165f - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene 9bcf4979-b04b-4bf3-8ad6-6e35e3be165f:
    - Deceleration due to nearby automobiles: expect 'decelerate' commitment
      and speed drop of at least 2.65 m/s, graded as 0.7 * min(1.0, drop / 5.3).
    """

    # Initialize component scores
    comp = {
        "decelerate_executed": 0.0,
    }

    # Check for deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop, considering timing
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s)) * traj.dt_s
        if min_speed_time > 3.0:  # Ensure deceleration happens later in the window
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 5.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
