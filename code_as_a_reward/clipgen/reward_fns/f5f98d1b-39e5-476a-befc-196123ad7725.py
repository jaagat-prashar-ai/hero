"""clip f5f98d1b-39e5-476a-befc-196123ad7725 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.31, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene: Deceleration to maintain a safe distance from a merging vehicle.
    Thresholds derived from expert trajectory: speed change, timing of deceleration.
    """
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_commitment": 0.0,
        "speed_change_execution": 0.0
    }

    # Perceptual mention of a vehicle
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1  # Small weight for mention

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2  # Weight for commitment presence

        # Trajectory execution: Speed change
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0.0, 6.4))
        speed_drop = initial_speed - min_speed_after
        
        # Graded factor for speed change
        comp["speed_change_execution"] = 0.7 * min(1.0, speed_drop / 1.0)  # Graded, with floor at half the scene's magnitude

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
