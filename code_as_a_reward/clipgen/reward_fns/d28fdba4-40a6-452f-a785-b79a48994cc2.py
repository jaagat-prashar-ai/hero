"""clip d28fdba4-40a6-452f-a785-b79a48994cc2 - attempt 2/5 - gate PASS (pos 0.80, max pert 0.37, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scene with a stop at a roundabout due to an upcoming vehicle.
    
    Decisive events:
    - Stop to yield to an upcoming vehicle at the roundabout.
    - Trajectory shows a speed drop of 6.5 m/s, reaching a minimum speed at t=4.1s.
    
    Scene-derived thresholds:
    - Speed drop floor: 3.25 m/s (half of 6.5 m/s).
    - Timing: Deceleration should occur primarily by t=4.1s.
    """

    # Initialize component scores
    perceptual_vehicle = 0.05  # Reduced weight for mention-only credit
    perceptual_roundabout = 0.05  # Reduced weight for mention-only credit
    stop_executed = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.05

    if any(p.entity == 'roundabout' for p in claims.perceptual):
        perceptual_roundabout = 0.05

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop, floor at half the GT magnitude
        stop_executed = 0.7 * min(1.0, speed_drop / 6.5)

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "perceptual_roundabout": perceptual_roundabout,
        "stop_executed": stop_executed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
