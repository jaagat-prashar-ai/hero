"""clip 0af40d4b-4c73-4431-ab3b-79acbe30338e - attempt 1/5 - gate PASS (pos 1.00, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scoring the rollout based on the decisive event of turning left through a construction zone.
    
    Decisive Event:
    - Left turn through construction zone, following temporary lane delineated by traffic cones.
    
    Scene-Derived Thresholds:
    - Lateral offset change: at least +4.7 meters (half of +9.46 meters)
    - Heading change: at least +5.6 degrees (half of +11.2 degrees)
    - Speed drop: at least 1.4 m/s (half of 2.8 m/s)
    """
    perceptual_weight = 0.1
    lateral_weight = 0.5
    speed_weight = 0.4

    # Perceptual component: mention of construction-related entities
    saw_construction = perceptual_weight * any(
        p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers')
        for p in claims.perceptual
    )

    # Lateral commitment and trajectory component
    lateral_commitment = any(
        c.maneuver in ('turn', 'nudge', 'merge', 'lane_change', 'enter', 'exit') and c.direction != 'right'
        for c in claims.commitments
    )
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    lateral_factor = lateral_weight * min(1.0, max(0.0, (lateral_offset_change - 4.7) / 4.76)) if lateral_commitment else 0.0

    # Heading change component
    heading_change = traj.total_heading_change_deg
    heading_factor = lateral_weight * min(1.0, max(0.0, (heading_change - 5.6) / 5.6)) if lateral_commitment else 0.0

    # Speed commitment and trajectory component
    speed_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    speed_factor = speed_weight * min(1.0, max(0.0, (speed_drop - 1.4) / 1.4)) if speed_commitment else 0.0

    return {
        "saw_construction": saw_construction,
        "lateral_factor": lateral_factor,
        "heading_factor": heading_factor,
        "speed_factor": speed_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
