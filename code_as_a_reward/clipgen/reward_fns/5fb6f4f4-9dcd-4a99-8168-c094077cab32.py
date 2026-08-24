"""clip 5fb6f4f4-9dcd-4a99-8168-c094077cab32 - attempt 1/5 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with deceleration to maintain safe distance and lateral stability.
    
    Decisive Events:
    1. Deceleration to maintain a safe distance from the vehicle ahead.
       - Perceptual: 'vehicle_generic'
       - Commitment: 'decelerate' (speed_profile)
       - Trajectory: Speed drop of at least 0.05 m/s, ideally around 0.1 m/s.
    2. Lateral stability against trailers on the right.
       - Perceptual: 'vehicle_generic' (for trailers)
       - No specific commitment needed for lateral stability.
       - Trajectory: Maintain lateral offset within ±0.03 m.
    """
    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_commitment": 0.0,
        "speed_drop_execution": 0.0,
        "lateral_stability": 0.0
    }
    
    # Perceptual component for vehicle mention
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Commitment component for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["decelerate_commitment"] = 0.2

        # Trajectory component for speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        if speed_drop >= 0.05:
            comp["speed_drop_execution"] = 0.5 * min(1.0, speed_drop / 0.1)

    # Trajectory component for lateral stability
    max_lateral_offset = max(abs(traj.final_lateral_offset_m), abs(traj.final_lateral_offset_m))
    if max_lateral_offset <= 0.03:
        comp["lateral_stability"] = 0.2

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
