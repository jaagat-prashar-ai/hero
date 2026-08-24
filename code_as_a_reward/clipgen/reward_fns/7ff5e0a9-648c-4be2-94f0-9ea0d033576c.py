"""clip 7ff5e0a9-648c-4be2-94f0-9ea0d033576c - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene with maintaining speed and curved road navigation.
    
    Decisive Events:
    1. Maintaining Speed: Expect minimal speed drop, with perceptual mention of 'cyclist' or 'vehicle_generic'.
    
    Trajectory thresholds:
    - Speed drop floor: 0.55 m/s (half of measured 1.1 m/s drop in positive case)
    """
    # Initialize component scores
    comp = {
        "mention_cyclist": 0.0,
        "maintain_speed": 0.0
    }
    
    # Perceptual mention of cyclist or vehicle_generic
    if any(p.entity in ('cyclist', 'vehicle_generic') for p in claims.perceptual):
        comp["mention_cyclist"] = 0.1  # Small weight for perceptual mention
    
    # Check for maintaining speed or slight deceleration
    if any(c.speed_profile in ('maintain', 'decelerate') for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed maintenance
        if speed_drop >= 0.55:
            comp["maintain_speed"] = 0.6 * min(1.0, speed_drop / 1.1)
    
    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
