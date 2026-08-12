"""clip 0dd725e0-11c1-47b8-b061-e64198785267 - attempt 2/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to pedestrians.
    The scene-derived thresholds are based on the expert's behavior of reducing speed significantly
    in response to a pedestrian crossing the crosswalk.
    """

    # Initialize component scores
    saw_pedestrian = 0.0
    saw_crosswalk = 0.0
    committed_to_yield = 0.0
    executed_yield = 0.0

    # Check perceptual claims
    for claim in claims.perceptual:
        if claim.entity == 'pedestrian' and claim.state == 'crossing':
            saw_pedestrian = 0.2
        if claim.entity == 'crosswalk' and claim.state == 'crossing':
            saw_crosswalk = 0.1

    # Check commitment claims
    for commitment in claims.commitments:
        if commitment.maneuver == 'yield' and commitment.speed_profile == 'decelerate':
            committed_to_yield = 0.2

    # Check trajectory execution
    if traj.n_waypoints > 0:
        # Speed reduction check
        speed_drop = traj.initial_speed_mps - traj.final_speed_mps
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if speed_drop >= 5.5 and len(speed_window) > 0 and np.min(speed_window) <= 1.5:
            # Require both a commitment claim and trajectory execution for full credit
            if committed_to_yield > 0:
                executed_yield = 0.5

    return {
        "saw_pedestrian": saw_pedestrian,
        "saw_crosswalk": saw_crosswalk,
        "committed_to_yield": committed_to_yield,
        "executed_yield": executed_yield
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
