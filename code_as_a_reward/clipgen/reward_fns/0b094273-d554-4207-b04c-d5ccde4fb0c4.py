"""clip 0b094273-d554-4207-b04c-d5ccde4fb0c4 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.40, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Detect pedestrians and crosswalk.
    - Commit to yielding by decelerating.
    - Execute a significant speed reduction within the rollout horizon, with correct timing, only if a commitment to yield is present.
    """

    # Initialize component scores
    saw_pedestrians = 0.0
    saw_crosswalk = 0.0
    committed_to_yield = 0.0
    executed_deceleration = 0.0

    # Check perceptual claims
    for perceptual_claim in claims.perceptual:
        if perceptual_claim.entity == 'pedestrian':
            saw_pedestrians = 0.1
        if perceptual_claim.entity == 'crosswalk':
            saw_crosswalk = 0.1

    # Check commitment claims
    has_yield_commitment = False
    for commitment_claim in claims.commitments:
        if (commitment_claim.maneuver == 'yield' and
            commitment_claim.speed_profile == 'decelerate'):
            committed_to_yield = 0.2
            has_yield_commitment = True

    # Check trajectory execution
    if traj.n_waypoints > 0 and has_yield_commitment:
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Check if the speed drop is significant and occurs at the correct time
        min_speed_time = np.argmin(traj.speed_mps) * traj.dt_s
        if speed_drop >= 3.0 and 6.0 <= min_speed_time <= 6.4:
            executed_deceleration = 0.6

    return {
        "saw_pedestrians": saw_pedestrians,
        "saw_crosswalk": saw_crosswalk,
        "committed_to_yield": committed_to_yield,
        "executed_deceleration": executed_deceleration
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
