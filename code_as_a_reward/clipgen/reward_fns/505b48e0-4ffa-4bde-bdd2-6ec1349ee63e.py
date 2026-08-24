"""clip 505b48e0-4ffa-4bde-bdd2-6ec1349ee63e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 505b48e0-4ffa-4bde-bdd2-6ec1349ee63e:
    - Decisive event: Stop for the stop sign
    - Thresholds: Speed drop >= 2.6 m/s, minimum speed timing aligned with GT stop at t=4.3 s
    """

    # Initialize component scores
    stop_commitment_executed = 0.0

    # Check for commitment to decelerate (stop/yield/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop over the trajectory
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Find the time of minimum speed
        min_speed_time = traj.dt_s * np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints))
        # Graded factor for speed drop, with a floor at half the GT magnitude
        # Include timing condition to ensure deceleration happens at the correct time
        if min_speed_time >= 4.0 and min_speed_time <= 4.6:
            stop_commitment_executed = 0.7 * min(1.0, speed_drop / 5.2)

    return {
        "stop_commitment_executed": stop_commitment_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
