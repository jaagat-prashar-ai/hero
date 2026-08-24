"""clip c11a87ef-d112-482a-8eb2-ee6610507b52 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene c11a87ef-d112-482a-8eb2-ee6610507b52:
    - Acceleration after clearing the intersection
    Trajectory thresholds:
    - Speed increase >= 2.0 m/s for acceleration
    """
    comp = {
        "acceleration_executed": 0.0,
    }

    # Acceleration commitment and execution
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        speed_increase = traj.final_speed_mps - traj.min_speed_mps
        if speed_increase >= 2.0:
            comp["acceleration_executed"] = 0.7 * min(1.0, speed_increase / 3.4)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
