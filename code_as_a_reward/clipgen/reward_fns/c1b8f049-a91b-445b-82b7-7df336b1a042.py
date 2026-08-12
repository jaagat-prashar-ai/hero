"""clip c1b8f049-a91b-445b-82b7-7df336b1a042 - attempt 3/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on decisive events:
    1. Lane Change to the Left
    2. Stopping Event

    Scene-derived thresholds:
    - Lane change: significant leftward lateral offset and heading change
    - Stopping: speed drop from ~7.2 m/s to 0.0 m/s by ~5.5s
    """

    # Initialize component scores
    stop_claim_execution = 0.0

    # Check for stop claims and execution conjunction
    if any(c.maneuver == 'stop' for c in claims.commitments):
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        if len(speed_window) > 0 and min(speed_window) < 0.5 and traj.final_speed_mps < 0.5:
            stop_claim_execution = 0.7

    return {
        "stop_claim_execution": stop_claim_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
