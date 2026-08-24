"""clip 7f008442-8fe4-4f34-afc1-7eaf5ef95c68 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.40, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Deceleration to yield to a pedestrian: Expect a mention of a pedestrian and a deceleration commitment.
    - Lane keeping amidst nearby vehicles: Expect a mention of vehicles, with stable lateral offset.
    - Trajectory thresholds are based on half the magnitude of the GT scene.
    """

    # Initialize component scores
    components = {
        "mention_pedestrian": 0.0,
        "decelerate_for_pedestrian": 0.0,
        "mention_vehicle": 0.0,
        "stable_lane_keeping": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        components["mention_pedestrian"] = 0.1

    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        components["mention_vehicle"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration
        components["decelerate_for_pedestrian"] = 0.5 * min(1.0, speed_drop / 3.0)

    # Check for stable lane keeping
    lateral_offset_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
    if lateral_offset_change <= 0.5:  # Allow some lateral movement
        components["stable_lane_keeping"] = 0.3

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
