"""clip 9da665bf-2834-489c-ae3e-877cf41023dc - attempt 2/5 - gate PASS (pos 1.00, max pert 0.30, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with oncoming vehicle turning left.
    
    Decisive Event:
    1. Oncoming vehicle turning left: Decelerate to maintain safe distance.
       - Perceptual: 'vehicle_generic', 'oncoming_traffic'
       - Commitment: 'decelerate' (speed_profile)
       - Trajectory: Speed drop >= 0.4 m/s, graded factor
    """
    comp = {
        "decelerate_execution": 0.0,
    }

    # Commitment and trajectory component for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 3.0, 6.3))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop, conditioned on commitment
        comp["decelerate_execution"] = 1.0 * min(1.0, speed_drop / 4.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
