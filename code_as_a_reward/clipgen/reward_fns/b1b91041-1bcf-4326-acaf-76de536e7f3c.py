"""clip b1b91041-1bcf-4326-acaf-76de536e7f3c - attempt 5/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene b1b91041-1bcf-4326-acaf-76de536e7f3c:
    - Proximity to Automobiles: Expect a speed increase, with a floor of 3.25 m/s.
    """
    comp = {
        "speed_increase": 0.0,
    }

    # Trajectory analysis
    speed_increase = traj.final_speed_mps - traj.initial_speed_mps
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        if speed_increase > 3.25:
            comp["speed_increase"] = 1.0 * min(1.0, speed_increase / 6.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
