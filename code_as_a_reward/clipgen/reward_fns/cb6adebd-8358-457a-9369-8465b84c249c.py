"""clip cb6adebd-8358-457a-9369-8465b84c249c - attempt 3/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring the rollout based on decisive events:
    1. Yield to the yellow emergency vehicle.
    2. Initial speed reduction.
    Thresholds derived from the scene: speed drop of at least 1.5 m/s, speed reduction starting near t=0 and reaching a minimum by around t=2.5 s.
    """
    # Initialize component scores
    yield_claim_and_execution = 0.0
    speed_reduction_claim_and_execution = 0.0

    # Check for perceptual claim of emergency vehicle and commitment to yield
    if any(claim.entity == 'emergency_vehicle' for claim in claims.perceptual) and \
       any(claim.maneuver == 'yield' for claim in claims.commitments):
        # Check for speed reduction execution with timing
        if traj.n_waypoints > 0:
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            initial_speed = traj.initial_speed_mps
            min_speed = min(speed_window) if len(speed_window) > 0 else initial_speed
            speed_drop = initial_speed - min_speed
            min_speed_time = np.argmin(speed_window) * traj.dt_s if len(speed_window) > 0 else 0

            # Require both claim and execution for yielding with timing
            if speed_drop >= 1.5 and min_speed < initial_speed and min_speed_time <= 2.5:
                yield_claim_and_execution = 0.4

    # Check for speed reduction claim and execution with timing
    if any(claim.maneuver == 'decelerate' for claim in claims.commitments):
        if traj.n_waypoints > 0:
            speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
            initial_speed = traj.initial_speed_mps
            min_speed = min(speed_window) if len(speed_window) > 0 else initial_speed
            speed_drop = initial_speed - min_speed
            min_speed_time = np.argmin(speed_window) * traj.dt_s if len(speed_window) > 0 else 0

            # Require both claim and execution for speed reduction with timing
            if speed_drop >= 1.5 and min_speed < initial_speed and min_speed_time <= 2.5:
                speed_reduction_claim_and_execution = 0.3

    return {
        "yield_claim_and_execution": yield_claim_and_execution,
        "speed_reduction_claim_and_execution": speed_reduction_claim_and_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
