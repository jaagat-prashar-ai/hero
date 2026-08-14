"""clip 8a94d595-7546-4339-9efb-269581d4fcaf - attempt 1/5 - gate PASS (pos 0.95, max pert 0.45, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with strong deceleration for construction zone and large vehicle ahead.
    
    Decisive Events:
    1. Strong deceleration for construction zone and large vehicle ahead.
       - Perceptual entities: {'work_zone', 'construction_cones', 'vehicle_generic'}
       - Commitment family: 'decelerate'
       - Trajectory: Speed drop of at least 2.15 m/s by t=5.1 s.
    2. Presence of nearby automobiles (awareness, not a decisive maneuver).
       - Perceptual entities: {'vehicle_generic'}
       - No specific trajectory change required beyond deceleration.
    
    Trajectory thresholds are one-sided and graded, with a focus on speed drop.
    """

    # Initialize component scores
    comp = {
        "perceptual_construction_zone": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_deceleration": 0.0,
        "perceptual_nearby_vehicles": 0.0
    }

    # Perceptual claims
    if any(p.entity in {'work_zone', 'construction_cones', 'vehicle_generic'} for p in claims.perceptual):
        comp["perceptual_construction_zone"] = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["perceptual_nearby_vehicles"] = 0.05

    # Commitment claims
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.3

        # Trajectory check for deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 2.15:
            comp["trajectory_deceleration"] = 0.5 * min(1.0, speed_drop / 4.3)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
