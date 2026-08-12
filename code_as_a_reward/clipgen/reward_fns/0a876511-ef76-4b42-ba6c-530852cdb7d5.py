"""clip 0a876511-ef76-4b42-ba6c-530852cdb7d5 - attempt 2/3 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "stop_claim_and_execution": 0.0,
        "caution_claim_and_execution": 0.0
    }

    # Check for perceptual and commitment claims
    saw_red_light = any(pc.entity == 'signal' and pc.state == 'red' for pc in claims.perceptual)
    committed_to_stop = any(cc.maneuver == 'stop' for cc in claims.commitments)

    # Check trajectory for deceleration
    speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    min_speed_time = np.argmin(speed_window) * traj.dt_s

    # Conjunction: Stop claim and execution
    if saw_red_light and committed_to_stop and (initial_speed - min_speed) >= 13.0 and min_speed_time >= 6.0:
        scores["stop_claim_and_execution"] = 0.7

    # Conjunction: Caution claim and execution
    if committed_to_stop and (initial_speed - min_speed) >= 13.0 and min_speed_time >= 6.0:
        scores["caution_claim_and_execution"] = 0.3

    return scores

def reward(claims, traj):
    """Decisive events: Red traffic light (stop).
    Thresholds: Deceleration >= 13.0 m/s, Min speed time >= 6.0s."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
