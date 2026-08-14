"""clip db61965a-ac0a-41b1-9714-decdcedbaf34 - attempt 1/5 - gate PASS (pos 0.95, max pert 0.45, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with pedestrians at crosswalk and nearby vehicles.
    
    Decisive Events:
    1. Yield to crossing pedestrians at the crosswalk.
       - Perceptual: Mention of 'pedestrian' or 'crosswalk'.
       - Commitment: Deceleration (speed_profile='decelerate').
       - Trajectory: Speed drop of at least 1.5 m/s, graded with execution quality.
    2. Presence of nearby vehicles.
       - Perceptual: Mention of 'vehicle_generic'.
       - No specific commitment required; focus on lateral stability.
       - Trajectory: Maintain lateral offset within ±0.54 m.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0,
        "perceptual_vehicle": 0.0,
        "trajectory_lateral_stability": 0.0,
    }

    # Perceptual: Pedestrian or Crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment: Decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.2

        # Trajectory: Deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.5:
            comp["trajectory_decelerate"] = 0.5 * min(1.0, speed_drop / 2.9)

    # Perceptual: Vehicle
    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    # Trajectory: Lateral Stability
    lateral_stability = max(abs(traj.final_lateral_offset_m), abs(traj.lateral_offset_m[0]))
    if lateral_stability <= 0.54:
        comp["trajectory_lateral_stability"] = 0.15 * min(1.0, (0.54 - lateral_stability) / 0.54)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
