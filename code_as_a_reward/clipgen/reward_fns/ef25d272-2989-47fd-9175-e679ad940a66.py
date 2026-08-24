"""clip ef25d272-2989-47fd-9175-e679ad940a66 - attempt 3/5 - gate PASS (pos 0.84, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the construction zone navigation scene.
    
    Decisive Event: Construction Zone Navigation
    - Commitment to maintain or slightly decelerate speed.
    - Trajectory should show a speed drop of at least 0.15 m/s and a heading change of at least +0.5 degrees.
    """
    commitment_credit = 0.0
    trajectory_credit = 0.0

    # Check for commitment claims related to maintaining or decelerating speed
    if any(c.speed_profile in ('maintain', 'decelerate') for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed = np.min(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed

        # Calculate heading change
        total_heading_change = traj.total_heading_change_deg

        # Graded trajectory factor for speed drop
        speed_drop_factor = 0.5 * min(1.0, speed_drop / 0.3)

        # Graded trajectory factor for heading change
        heading_change_factor = 0.5 * min(1.0, total_heading_change / 1.0)

        # Combine trajectory factors
        trajectory_credit = 0.55 * speed_drop_factor + 0.35 * heading_change_factor

        # Ensure commitment credit is given if the trajectory supports it
        if speed_drop >= 0.15 and total_heading_change >= 0.5:
            commitment_credit = 0.45

    return {
        "commitment_execution": commitment_credit,
        "trajectory_execution": trajectory_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
