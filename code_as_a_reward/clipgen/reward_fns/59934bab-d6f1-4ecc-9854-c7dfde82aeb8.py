"""clip 59934bab-d6f1-4ecc-9854-c7dfde82aeb8 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene 59934bab-d6f1-4ecc-9854-c7dfde82aeb8:
    - Decisive Event: Adapt speed for road curvature
    - Commitment: 'decelerate' family
    - Trajectory: Speed reduction of at least 4.25 m/s by t=6.1 s, graded
    """
    comp = {
        "decelerate_executed": 0.0,
    }

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded trajectory factor for speed reduction
        if speed_drop >= 4.25:
            comp["decelerate_executed"] = 0.7 * min(1.0, speed_drop / 8.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
