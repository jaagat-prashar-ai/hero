"""clip 9a51dbf3-262e-4f46-8472-9ff1c31fe29b - attempt 2/5 - gate PASS (pos 0.73, max pert 0.18, real rollout argmax 6)"""
def components(claims, traj):
    """Components for navigating a construction zone with a rightward maneuver and deceleration.
    
    Decisive Events:
    1. Navigating the construction zone marked by traffic cones, requiring a rightward maneuver and deceleration.
    
    Scene-Derived Thresholds:
    - Speed drop: at least 2.1 m/s (half of 4.2 m/s)
    - Heading change: at least -10.2 degrees (half of -20.4 degrees)
    - Perceptual mention of construction-related entities
    """
    comp = {
        "perceptual_construction": 0.05,  # Reduced weight for mention-only credit
        "decelerate_and_maneuver_right": 0.0
    }

    # Perceptual component: mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.05

    # Commitment component: decelerate and maneuver right
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Speed drop factor
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        speed_factor = 0.7 * min(1.0, speed_drop / 6.0)

        # Heading change factor
        heading_change = traj.total_heading_change_deg
        heading_factor = 0.3 * min(1.0, abs(heading_change) / 20.4)

        # Check for rightward maneuver
        if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') and c.direction != 'left' for c in claims.commitments):
            comp["decelerate_and_maneuver_right"] = speed_factor + heading_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
