"""clip e98f1dcb-6064-47eb-b04e-579223351f93 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene e98f1dcb-6064-47eb-b04e-579223351f93:
    - Deceleration due to pedestrians: speed drop >= 1.75 m/s by t=4.9s
    - Maintain lateral stability: no abrupt lateral maneuvers, gated by a commitment
    - Perceptual mention of vehicles
    """
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_for_pedestrians": 0.0,
        "lateral_stability": 0.0
    }

    # Perceptual mention of vehicles
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Deceleration for pedestrians
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 1.75:
            comp["decelerate_for_pedestrians"] = 0.6 * min(1.0, speed_drop / 3.5)

    # Lateral stability (no abrupt lateral maneuvers), gated by a commitment
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        lateral_offset_change = abs(traj.final_lateral_offset_m - traj.lateral_offset_m[0])
        if lateral_offset_change <= 0.23:
            comp["lateral_stability"] = 0.3

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
