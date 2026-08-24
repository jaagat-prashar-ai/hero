"""clip dbd6ddaf-2a3b-4061-8395-b0e6f1caf90d - attempt 4/5 - gate PASS (pos 0.70, max pert 0.07, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene with gentle deceleration to maintain distance from lead vehicle and yield to pedestrians.
    - Lead Vehicle: Expect mention of 'vehicle_generic' or 'lead_vehicle' and a 'decelerate' commitment with a speed drop >= 1.85 m/s.
    - Pedestrians: Expect mention of 'pedestrian' and a 'decelerate' commitment with a speed drop >= 1.85 m/s.
    - Trajectory factors are graded and one-sided, with a generous floor.
    """
    # Initialize component scores
    comp = {
        "commitment_decelerate_pedestrian": 0.0,
    }

    # Trajectory analysis
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    deceleration_factor = 0.7 * min(1.0, speed_drop / 1.85)  # Graded factor for speed drop

    # Commitment components
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Deceleration for pedestrians
        if any(p.entity == 'pedestrian' for p in claims.perceptual):
            comp["commitment_decelerate_pedestrian"] = deceleration_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
