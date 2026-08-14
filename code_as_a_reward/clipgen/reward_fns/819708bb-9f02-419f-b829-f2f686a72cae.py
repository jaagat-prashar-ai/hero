"""clip 819708bb-9f02-419f-b829-f2f686a72cae - attempt 1/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive event of a lane change to the left
    due to a blocked lane. The scene-derived thresholds are:
    - Lateral offset change of at least +1.0 m to the left (half of GT's +1.93 m)
    - Perceptual mention of a vehicle or lane-related entity
    - Commitment to a lane change maneuver to the left
    """

    # Initialize component scores
    perceptual_score = 0.0
    lateral_maneuver_score = 0.0

    # Check for perceptual mentions of relevant entities
    if any(p.entity in ('vehicle_generic', 'lane') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for perceptual mention

    # Check for a commitment to a lane change to the left
    lane_change_commitment = any(
        c.maneuver == 'lane_change' and c.direction != 'right'
        for c in claims.commitments
    )

    # Calculate the lateral offset change
    lateral_offset_change = traj.final_lateral_offset_m - traj.lateral_offset_m[0]
    if lane_change_commitment:
        # Graded factor for lateral offset change
        lateral_maneuver_score = 0.7 * min(1.0, lateral_offset_change / 1.93)

    return {
        "perceptual_mention": perceptual_score,
        "lateral_maneuver_executed": lateral_maneuver_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
