"""clip 4befabdb-4376-40e8-8561-df5e0aa12b67 - attempt 4/5 - gate PASS (pos 1.00, max pert 0.55, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scene with a red traffic light:
    - Deceleration commitment for the red traffic light.
    - Perceptual mention of the traffic light.
    - Trajectory shows significant speed drop indicating a stop.
    """
    comp = {
        "perceptual_mention_signal": 0.05,
        "commitment_decelerate": 0.0,
        "trajectory_stop_execution": 0.0,
    }

    # Perceptual mention of the traffic light
    if any(p.entity == 'signal' for p in claims.perceptual):
        comp["perceptual_mention_signal"] = 0.05

    # Calculate speed drop
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    speed_drop = initial_speed - min_speed

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Graded factor for speed drop indicating a stop
        if speed_drop >= 5.0:  # Expecting a significant deceleration
            comp["commitment_decelerate"] = 0.45 * min(1.0, speed_drop / 10.0)

    # Trajectory expectation: significant speed drop
    if traj.stop_event:
        comp["trajectory_stop_execution"] = 0.5 * min(1.0, speed_drop / 10.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
