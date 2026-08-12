"""clip 8260e7c4-d797-4920-a98e-33e62158a7fd - attempt 3/3 - gate PASS (pos 0.80, max pert 0.20, real rollout argmax 0)"""
def components(claims, traj):
    # Initialize component scores
    comp = {
        "saw_red_light": 0.0,
        "commit_and_execute_stop": 0.0
    }

    # Check for perceptual claim of seeing a red traffic light
    if any(p.entity == 'signal' and p.state == 'red' for p in claims.perceptual):
        comp["saw_red_light"] = 0.2

    # Check for both commitment to stop and executed stop in the trajectory
    if any(c.maneuver == 'stop' and c.speed_profile == 'decelerate' for c in claims.commitments):
        if traj.n_waypoints > 0:
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            initial_speed = traj.initial_speed_mps
            final_speed = traj.final_speed_mps
            min_speed = traj.min_speed_mps

            # Check if the speed drops significantly and reaches near zero
            if initial_speed - final_speed >= 8.0 and min_speed <= 0.5:
                comp["commit_and_execute_stop"] = 0.6

    return comp

def reward(claims, traj):
    """
    Reward function for the scene involving stopping for a red traffic light.
    Decisive events include detecting the red light, committing to stop, and
    executing the stop with a significant speed drop.
    """
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
