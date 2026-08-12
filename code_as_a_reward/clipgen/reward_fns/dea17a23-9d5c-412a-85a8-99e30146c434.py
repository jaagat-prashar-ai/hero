"""clip dea17a23-9d5c-412a-85a8-99e30146c434 - attempt 2/3 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 9)"""
def components(claims, traj):
    """Components for reward function based on the decisive event of decelerating to maintain a safe distance from a vehicle ahead."""
    
    # Initialize component scores
    commitment_claim_score = 0.0
    trajectory_execution_score = 0.0
    conjunction_score = 0.0

    # Check for commitment claim: commitment to decelerate
    has_commitment_claim = any(
        commitment_claim.maneuver == 'decelerate' and commitment_claim.speed_profile == 'decelerate'
        for commitment_claim in claims.commitments
    )
    if has_commitment_claim:
        commitment_claim_score = 0.2

    # Check trajectory execution: significant deceleration with correct timing
    if traj.n_waypoints > 0:
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        min_speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4).min()
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        
        # Check if the speed drop is significant and occurs at the correct time
        if speed_drop >= 4.5 and min_speed_window <= 0.5 and 3.5 <= min_speed_time <= 5.0:
            trajectory_execution_score = 0.3

    # Conjunction: both claim and trajectory execution must be present
    if has_commitment_claim and trajectory_execution_score > 0:
        conjunction_score = 0.5

    return {
        "commitment_claim": commitment_claim_score,
        "trajectory_execution": trajectory_execution_score,
        "conjunction": conjunction_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
