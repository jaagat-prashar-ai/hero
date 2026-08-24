"""clip 9d7ba087-0750-43f8-9ebe-2823f5244079 - attempt 5/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene with deceleration to pass oncoming vehicle on narrow road.
    
    Decisive Events:
    1. Deceleration and Stop: Decelerate to pass the oncoming vehicle.
       - Perceptual: 'vehicle_generic', 'oncoming_traffic'
       - Commitment: 'decelerate' (speed_profile)
       - Trajectory: Speed drop >= 1.25 m/s within 6.3 s, graded factor
    
    2. Proximity to Oncoming Vehicle: Influence of oncoming vehicle on deceleration.
       - Perceptual: 'vehicle_generic', 'oncoming_traffic'
       - Commitment: 'decelerate' (speed_profile)
       - Trajectory: Same as above, as the vehicle presence influences deceleration.
    """
    # Rebudget the weights to sum to exactly 1.0
    perceptual_weight = 0.1
    commitment_weight = 0.4
    trajectory_weight = 0.5

    # Perceptual component
    saw_vehicle = any(p.entity in ('vehicle_generic', 'oncoming_traffic') for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_vehicle else 0.0

    # Commitment component
    committed_to_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    
    # Trajectory component
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed = traj.min_speed_mps
    speed_drop = initial_speed - min_speed

    # Graded trajectory factor for deceleration
    trajectory_score = trajectory_weight * min(1.0, speed_drop / 1.25) if committed_to_decelerate else 0.0

    return {
        "perceptual_vehicle": perceptual_score,
        "commitment_decelerate": commitment_weight if committed_to_decelerate else 0.0,
        "trajectory_decelerate": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
