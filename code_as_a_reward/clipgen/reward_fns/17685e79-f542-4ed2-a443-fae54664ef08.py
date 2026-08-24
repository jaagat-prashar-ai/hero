"""clip 17685e79-f542-4ed2-a443-fae54664ef08 - attempt 2/5 - gate PASS (pos 1.00, max pert 0.60, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Maintain speed: Check for speed maintenance commitment and trajectory.
    - Perceptual mention: Check for mention of road-related entities.
    - Road curvature: Check for trajectory alignment with road curvature.
    
    Scene-derived thresholds:
    - Speed maintenance: Speed drop <= 0.05 m/s.
    - Road curvature: Total heading change >= 32.8 degrees.
    """
    # Initialize component scores
    maintain_speed_score = 0.0
    perceptual_mention_score = 0.0
    road_curvature_score = 0.0

    # Check for perceptual mentions of road-related entities
    if any(p.entity in ('lane', 'curve') for p in claims.perceptual):
        perceptual_mention_score = 0.1

    # Check for speed maintenance commitment
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded score for maintaining speed
        maintain_speed_score = 0.5 * min(1.0, (0.05 - speed_drop) / 0.05)

    # Check for road curvature in trajectory
    if traj.total_heading_change_deg >= 32.8:
        # Graded score for road curvature alignment
        road_curvature_score = 0.4 * min(1.0, traj.total_heading_change_deg / 65.6)

    return {
        "maintain_speed": maintain_speed_score,
        "perceptual_mention": perceptual_mention_score,
        "road_curvature": road_curvature_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
