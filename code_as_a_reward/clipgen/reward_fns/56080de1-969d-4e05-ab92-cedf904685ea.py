"""clip 56080de1-969d-4e05-ab92-cedf904685ea - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing.
    
    Decisive Events:
    1. Pedestrian Crossing: Requires deceleration commitment and pedestrian mention.
       - Speed drop threshold: 2.5 m/s (half of GT's 5.1 m/s drop).
    
    Thresholds derived from GT trajectory: speed drop 5.1 m/s, min speed 4.3 m/s at t=5.3s.
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1

    # Check for deceleration commitment and matching trajectory execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for deceleration
        if speed_drop >= 2.5:
            comp["decelerate_for_pedestrian"] = 0.6 * min(1.0, speed_drop / 5.1)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
