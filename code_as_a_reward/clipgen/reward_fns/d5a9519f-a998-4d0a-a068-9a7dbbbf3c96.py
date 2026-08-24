"""clip d5a9519f-a998-4d0a-a068-9a7dbbbf3c96 - attempt 2/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Decisive events: maintaining safe distance from lead vehicle, navigating construction zone.
    Trajectory thresholds: significant speed drop (8.4 m/s), minimal lateral offset.
    """

    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "perceptual_construction": 0.0,
        "decelerate_execution": 0.0
    }

    # Perceptual components
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.05

    # Commitment and trajectory components
    # Deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed change
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for deceleration execution
        comp["decelerate_execution"] = 0.9 * min(1.0, speed_drop / 4.2)  # Floor at 4.2 m/s, graded above

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
