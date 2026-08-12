"""clip 29a6785d-0a1a-450d-87ff-e908d6b2120a - attempt 1/3 - gate PASS (pos 0.80, max pert 0.40, real rollout argmax 3)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "saw_pedestrian": 0.0,
        "commit_to_yield": 0.0,
        "commit_to_decelerate": 0.0,
        "executed_yield": 0.0,
        "executed_deceleration": 0.0
    }

    # Check perceptual claims for pedestrian
    if any(pc.entity == "pedestrian" and pc.state == "crossing" for pc in claims.perceptual):
        scores["saw_pedestrian"] = 0.2

    # Check commitment claims for yield and deceleration
    if any(cc.maneuver == "yield" for cc in claims.commitments):
        scores["commit_to_yield"] = 0.2
    if any(cc.maneuver == "decelerate" for cc in claims.commitments):
        scores["commit_to_decelerate"] = 0.2

    # Check trajectory for execution of yield (stop event)
    if traj.stop_event:
        scores["executed_yield"] = 0.2

    # Check trajectory for execution of deceleration
    if traj.initial_speed_mps > traj.final_speed_mps:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        min_speed = np.min(speed_window)
        if min_speed <= 0.3 and traj.final_speed_mps <= 0.3:
            scores["executed_deceleration"] = 0.2

    return scores

def reward(claims, traj):
    """Reward function for yielding to a pedestrian by decelerating.
    
    Decisive events:
    1. Detecting a pedestrian crossing.
    2. Committing to yield and decelerate.
    3. Executing a deceleration to a near stop within the 6.4s horizon.
    
    Thresholds:
    - Speed should drop from initial to near stop (<= 0.3 m/s).
    - Stop event should be true for yielding.
    - Perceptual and commitment claims must align with trajectory actions.
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
