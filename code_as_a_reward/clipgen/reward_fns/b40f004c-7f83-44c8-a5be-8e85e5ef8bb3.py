"""clip b40f004c-7f83-44c8-a5be-8e85e5ef8bb3 - attempt 2/3 - gate PASS (pos 0.90, max pert 0.50, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to pedestrians.
    Key checks include perceptual claims about pedestrians, commitment to yield, and trajectory
    execution showing speed reduction and lateral offset adjustment.
    """
    # Initialize component scores
    score_perceptual = 0.0
    score_commitment = 0.0
    score_trajectory = 0.0

    # Check for perceptual claims about pedestrians crossing
    if any(pc.entity == 'pedestrian' and pc.state == 'crossing' for pc in claims.perceptual):
        score_perceptual = 0.1

    # Check for commitment to yield
    if any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments):
        score_commitment = 0.2

    # Check trajectory for speed reduction and timing
    if traj.n_waypoints > 0:
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        final_speed = traj.final_speed_mps

        # Speed should drop significantly, similar to GT but with tolerance
        # Check that the minimum speed occurs around the expected time
        min_speed_time = np.argmin(traj.speed_mps) * traj.dt_s
        if initial_speed > 4.5 and min_speed < 2.5 and final_speed < initial_speed and 3.0 <= min_speed_time <= 4.5:
            score_trajectory += 0.4

        # Check for lateral offset adjustment
        final_lateral_offset = traj.final_lateral_offset_m
        if -2.5 <= final_lateral_offset <= -2.0:
            score_trajectory += 0.2

    # Conjunction: Require both a claim and matching trajectory execution
    if score_commitment > 0 and score_trajectory > 0:
        score_trajectory += 0.2

    return {
        "perceptual_claims": score_perceptual,
        "commitment_claims": score_commitment,
        "trajectory_execution": score_trajectory
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
