"""clip 40f66d6f-dbef-48d3-b5b7-2e185fb284a1 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Decisive events: deceleration behind lead vehicle (lead truck), with lateral stability.
    Trajectory thresholds: speed drop >= 2.05 m/s, lateral offset within ±0.15 m.
    """

    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "deceleration_executed": 0.0,
        "lateral_stability": 0.0
    }

    # Perceptual claim for lead vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.1

    # Commitment claim for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        comp["deceleration_executed"] = 0.6 * min(1.0, speed_drop / 4.1)

    # Lateral stability with commitment check
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        lateral_offset = max(abs(traj.final_lateral_offset_m), max(abs(window(traj.lateral_offset_m, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))))
        comp["lateral_stability"] = 0.3 * min(1.0, (0.15 - lateral_offset) / 0.15)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
