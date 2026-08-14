"""clip 05ee29be-af7b-4fd6-853f-96a49f0d6bc7 - attempt 1/5 - gate PASS (pos 0.95, max pert 0.45, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene 05ee29be-af7b-4fd6-853f-96a49f0d6bc7:
    - Deceleration to yield to pedestrians: expect 'decelerate' commitment,
      mention of 'pedestrian', and speed drop of at least 3.0 m/s.
    - Proximity to automobiles: mention of 'vehicle_generic', maintain
      minimal lateral offset.
    """
    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_decelerate": 0.0,
        "perceptual_vehicle": 0.0,
        "trajectory_lateral_stability": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'cyclist') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.3

        # Trajectory factor for deceleration
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        comp["trajectory_decelerate"] = 0.5 * min(1.0, speed_drop / 6.0)

    # Trajectory factor for lateral stability
    max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
    comp["trajectory_lateral_stability"] = 0.05 * min(1.0, 0.04 / max_lateral_offset)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
