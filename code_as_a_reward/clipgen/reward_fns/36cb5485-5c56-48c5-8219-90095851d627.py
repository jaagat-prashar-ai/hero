"""clip 36cb5485-5c56-48c5-8219-90095851d627 - attempt 1/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 2)"""
def components(claims, traj):
    # Initialize component scores
    components = {
        "saw_pedestrian": 0.0,
        "saw_stop_sign": 0.0,
        "committed_to_stop": 0.0,
        "executed_stop": 0.0,
        "yielded_for_pedestrian": 0.0
    }

    # Check for perceptual claims
    saw_pedestrian = any(claim.entity == 'pedestrian' for claim in claims.perceptual)
    saw_stop_sign = any(claim.maneuver == 'stop' for claim in claims.commitments)

    # Check for commitment claims
    committed_to_stop = any(claim.maneuver == 'stop' for claim in claims.commitments)

    # Check trajectory for execution
    speed_drop = traj.initial_speed_mps - traj.final_speed_mps
    executed_stop = speed_drop >= 9.8  # Allowing some tolerance from the GT drop of 10.3 m/s
    final_speed_low = traj.final_speed_mps <= 1.1  # Allowing some tolerance from the GT final speed of 0.6 m/s

    # Check for yielding behavior
    yielded_for_pedestrian = final_speed_low and executed_stop

    # Assign scores based on checks
    if saw_pedestrian:
        components["saw_pedestrian"] = 0.2
    if saw_stop_sign:
        components["saw_stop_sign"] = 0.2
    if committed_to_stop:
        components["committed_to_stop"] = 0.2
    if executed_stop:
        components["executed_stop"] = 0.2
    if yielded_for_pedestrian:
        components["yielded_for_pedestrian"] = 0.2

    return components

def reward(claims, traj):
    """Reward function for scene with stop sign and pedestrian crossing.
    Decisive events: stop for stop sign, yield for pedestrian.
    Thresholds: speed drop >= 9.8 m/s, final speed <= 1.1 m/s."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
