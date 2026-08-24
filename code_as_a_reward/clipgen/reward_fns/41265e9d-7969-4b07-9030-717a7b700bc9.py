"""clip 41265e9d-7969-4b07-9030-717a7b700bc9 - attempt 1/5 - gate PASS (pos 0.90, max pert 0.41, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene 41265e9d-7969-4b07-9030-717a7b700bc9:
    - Decelerate for pedestrian crossing: speed drop >= 0.8 m/s
    - Maintain lane discipline around vehicle cluster
    - Perceptual mention of pedestrian or vehicles
    """
    # Initialize component scores
    comp = {
        "decelerate_for_pedestrian": 0.0,
        "maintain_lane_discipline": 0.0,
        "mention_pedestrian": 0.0,
        "mention_vehicles": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity in ('pedestrian', 'workers') for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.1

    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["mention_vehicles"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        comp["decelerate_for_pedestrian"] = 0.5 * min(1.0, speed_drop / 1.6)

    # Check for lane discipline (minimal lateral offset change)
    lateral_offset_change = abs(traj.final_lateral_offset_m)
    if lateral_offset_change <= 0.28:
        comp["maintain_lane_discipline"] = 0.3

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
