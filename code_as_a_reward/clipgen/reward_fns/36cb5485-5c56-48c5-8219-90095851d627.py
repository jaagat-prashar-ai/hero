"""clip 36cb5485-5c56-48c5-8219-90095851d627 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with deceleration for stop sign and lane maintenance.
    
    Decisive Events:
    1. Deceleration for Stop Sign: Expect a 'decelerate' commitment with a speed drop of at least 4.6 m/s.
    
    Thresholds:
    - Speed drop: 4.6 m/s minimum for deceleration component.
    """
    # Initialize component scores
    comp = {
        "perceptual_traffic_control": 0.0,
        "commitment_decelerate": 0.0
    }

    # Perceptual claim for traffic control
    if any(p.entity in ('signal', 'intersection', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_traffic_control"] = 0.1

    # Commitment claim for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop
        comp["commitment_decelerate"] = 0.7 * min(1.0, speed_drop / 9.2)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
