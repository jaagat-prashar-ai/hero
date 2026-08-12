"""clip 3820a10d-ede2-4e57-b226-5175f4911406 - attempt 3/3 - gate PASS (pos 0.80, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """Components for the scene where the ego vehicle must stop at a stop sign due to pedestrian presence."""
    
    # Initialize component scores
    saw_pedestrian = 0.0
    committed_to_stop = 0.0
    executed_stop = 0.0

    # Check for perceptual claims
    for perceptual in claims.perceptual:
        if perceptual.entity == 'pedestrian':
            saw_pedestrian = 0.2  # Assign partial credit for recognizing pedestrians

    # Check for commitment claims
    for commitment in claims.commitments:
        if commitment.maneuver == 'stop' and commitment.speed_profile == 'decelerate':
            committed_to_stop = 0.2  # Assign partial credit for committing to stop

    # Check trajectory execution with conjunction of claims
    if traj.n_waypoints > 0 and committed_to_stop > 0:
        # Calculate speed drop within the rollout horizon
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_drop = initial_speed - final_speed

        # Check if the speed drop is significant and within the expected range
        if speed_drop >= 4.5 and traj.min_speed_mps <= 2.0 and traj.min_speed_mps == traj.speed_mps[63]:  # Ensure the minimum speed occurs at the end
            executed_stop = 0.6  # Assign significant credit for executing the stop

    return {
        "saw_pedestrian": saw_pedestrian,
        "committed_to_stop": committed_to_stop,
        "executed_stop": executed_stop
    }

def reward(claims, traj):
    """Calculate the total reward as the clamped sum of components."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
