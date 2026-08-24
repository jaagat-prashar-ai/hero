"""clip 15fd727a-c454-45e7-a2bc-5aecea725ee1 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scene with a parked vehicle requiring a leftward nudge.
    
    Decisive event: Steer left to pass a parked vehicle occupying part of the lane.
    - Perceptual entity: 'vehicle_generic' or 'lane'
    - Commitment family: 'nudge' or 'lane_change' with direction not 'right'
    - Trajectory: Leftward heading change of at least -1.8 degrees
    """
    perceptual_weight = 0.1
    lateral_weight = 0.6

    # Perceptual component: mention of a vehicle or lane
    saw_vehicle_or_lane = any(
        p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual
    )
    perceptual_score = perceptual_weight if saw_vehicle_or_lane else 0.0

    # Commitment component: nudge or lane_change to the left
    committed_to_left_nudge = any(
        c.maneuver in ('nudge', 'lane_change') and c.direction != 'right'
        for c in claims.commitments
    )

    # Trajectory component: leftward heading change
    heading_change = traj.total_heading_change_deg
    lateral_factor = 0.0
    if committed_to_left_nudge:
        # Graded factor based on leftward heading change
        lateral_factor = lateral_weight * min(1.0, max(0.0, -heading_change / 3.6))

    return {
        "saw_vehicle_or_lane": perceptual_score,
        "left_nudge_executed": lateral_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
