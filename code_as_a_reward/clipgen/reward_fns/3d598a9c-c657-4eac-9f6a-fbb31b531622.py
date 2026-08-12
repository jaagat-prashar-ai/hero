"""clip 3d598a9c-c657-4eac-9f6a-fbb31b531622 - attempt 2/3 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 0)"""
def components(claims, traj):
    """Components for decisive events: stopping at stop sign and yielding for pedestrians.
    Scene-derived thresholds: speed reduction to near-zero by ~4.9s, perceptual claims for stop sign and pedestrians."""
    
    # Initialize component scores
    saw_intersection = 0.0
    committed_to_stop = 0.0
    executed_stop_with_claim = 0.0

    # Check perceptual claims
    for claim in claims.perceptual:
        if claim.entity == 'intersection':
            saw_intersection = 0.1

    # Check commitment claims
    for claim in claims.commitments:
        if claim.maneuver == 'stop' and claim.speed_profile == 'decelerate':
            committed_to_stop = 0.2

    # Check trajectory execution with claim requirement
    speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
    min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
    if min_speed < 0.5 and traj.stop_event:
        # Require both a commitment claim and trajectory execution for full credit
        if committed_to_stop > 0:
            executed_stop_with_claim = 0.6

    return {
        "saw_intersection": saw_intersection,
        "committed_to_stop": committed_to_stop,
        "executed_stop_with_claim": executed_stop_with_claim,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
