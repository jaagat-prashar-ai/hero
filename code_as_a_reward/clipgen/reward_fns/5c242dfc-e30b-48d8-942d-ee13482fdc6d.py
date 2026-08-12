"""clip 5c242dfc-e30b-48d8-942d-ee13482fdc6d - attempt 2/3 - gate PASS (pos 1.00, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of stopping for the red traffic light.
    - Perceptual recognition of the red traffic light.
    - Commitment to stop with matching trajectory execution.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_and_trajectory_score = 0.0

    # Check for perceptual claim of red traffic light
    for perceptual in claims.perceptual:
        if perceptual.entity == 'signal' and perceptual.state == 'red':
            perceptual_score = 0.2
            break

    # Check for commitment to stop and matching trajectory execution
    commitment_present = False
    for commitment in claims.commitments:
        if commitment.maneuver == 'stop' and commitment.speed_profile == 'decelerate':
            commitment_present = True
            break

    if commitment_present and traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        min_speed = traj.min_speed_mps

        # Check if the speed drops significantly
        if initial_speed > 8.0 and final_speed < 1.0 and min_speed < 1.0:
            commitment_and_trajectory_score = 0.8

    return {
        "perceptual_recognition": perceptual_score,
        "commitment_and_trajectory": commitment_and_trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
