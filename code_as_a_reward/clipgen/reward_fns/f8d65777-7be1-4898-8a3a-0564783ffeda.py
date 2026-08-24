"""clip f8d65777-7be1-4898-8a3a-0564783ffeda - attempt 1/5 - gate PASS (pos 0.90, max pert 0.30, real rollout argmax 2)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's response to navigating through a construction zone.
    Decisive event: navigating through construction zone with slight deceleration.
    Scene-derived thresholds: speed drop >= 0.5 m/s, lateral offset adjustment.
    """

    # Initialize component scores
    perceptual_construction_zone = 0.0
    commitment_decelerate = 0.0
    trajectory_decelerate = 0.0
    trajectory_lateral_adjustment = 0.0

    # Perceptual check for construction zone
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual):
        perceptual_construction_zone = 0.1

    # Commitment check for deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory check for speed reduction
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        trajectory_decelerate = 0.5 * min(1.0, speed_drop / 1.0)  # Graded factor based on speed drop
        commitment_decelerate = 0.3  # Base score for having the decelerate commitment

    # Trajectory check for lateral adjustment
    lateral_offset_change = abs(traj.final_lateral_offset_m)
    trajectory_lateral_adjustment = 0.1 * min(1.0, lateral_offset_change / 0.77)  # Graded factor

    # Return component scores
    return {
        "perceptual_construction_zone": perceptual_construction_zone,
        "commitment_decelerate": commitment_decelerate,
        "trajectory_decelerate": trajectory_decelerate,
        "trajectory_lateral_adjustment": trajectory_lateral_adjustment
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
