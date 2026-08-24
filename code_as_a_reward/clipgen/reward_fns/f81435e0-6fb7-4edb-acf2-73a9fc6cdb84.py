"""clip f81435e0-6fb7-4edb-acf2-73a9fc6cdb84 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for reward function based on decisive events:
    1. Pedestrian crossing: Expect deceleration to maintain safe distance.
       - Perceptual mention: 'pedestrian'
       - Commitment: 'decelerate' family
       - Trajectory: Speed drop of at least 0.5 m/s
    2. Automobile proximity: No significant action required.
       - Perceptual mention: 'vehicle_generic' (minimal weight)
       - No commitment or trajectory change expected
    """

    # Initialize component scores
    comp = {
        "saw_pedestrian": 0.0,
        "decelerate_executed": 0.0,
        "saw_vehicle": 0.0
    }

    # Check for perceptual mentions
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["saw_pedestrian"] = 0.1  # Small weight for mention

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["saw_vehicle"] = 0.05  # Minimal weight for mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for deceleration execution
        comp["decelerate_executed"] = 0.6 * min(1.0, speed_drop / 0.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
