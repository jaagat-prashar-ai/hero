"""clip 8843953f-cec8-43fe-9eba-4fa20afe5999 - attempt 2/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """Components for yielding to a pedestrian crossing at a crosswalk.
    
    Decisive event: Yield to pedestrian (track 112) crossing the road.
    Scene-derived thresholds: 
    - Speed reduction to below 1.5 m/s by around t=6.0 s.
    - Recognition of pedestrian and crosswalk.
    - Commitment to yield.
    """
    # Initialize component scores
    commitment_and_trajectory_score = 0.0

    # Check commitment claims
    committed_to_yield = any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments)

    # Check trajectory execution
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
        speed_reduction = traj.initial_speed_mps - min_speed
        min_speed_time = np.argmin(speed_window) * traj.dt_s if len(speed_window) > 0 else float('inf')

        # Check if the speed reduction is significant and occurs at the expected time
        if committed_to_yield and speed_reduction >= 3.5 and min_speed <= 1.5 and 5.0 <= min_speed_time <= 6.4:
            commitment_and_trajectory_score = 0.7

    # Return the component scores
    return {
        "commitment_and_trajectory_score": commitment_and_trajectory_score
    }

def reward(claims, traj):
    """Calculate the total reward based on component scores."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
