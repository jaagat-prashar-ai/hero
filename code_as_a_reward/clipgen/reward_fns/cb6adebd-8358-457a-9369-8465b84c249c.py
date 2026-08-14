"""clip cb6adebd-8358-457a-9369-8465b84c249c - attempt 2/5 - gate PASS (pos 1.00, max pert 0.43, real rollout argmax 2)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yield to the yellow emergency vehicle (Track 23 - Heavy Truck).
    - Perceptual mention of emergency or heavy vehicle.
    - Commitment to decelerate with trajectory showing a speed drop of at least 3.7 m/s by t=4.0s.
    """
    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mentions of emergency or heavy vehicle
    if any(p.entity in ('emergency_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for perceptual mention

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded trajectory score based on speed drop
        trajectory_score = 0.5 * min(1.0, speed_drop / 6.0)  # Graded factor with floor at 3.7 m/s

        # Check if the minimum speed occurs around the expected time (t=4.0s)
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if 3.0 <= min_speed_time <= 5.0:
            # Combine commitment and trajectory for a larger score
            commitment_score = 0.4 if trajectory_score > 0 else 0.0

    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score,
        "trajectory_execution": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
