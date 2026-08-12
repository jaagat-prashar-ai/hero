"""clip 03a83788-ce57-4bec-bf99-e6e82f64227a - attempt 2/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 7)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "saw_pedestrian": 0.0,
        "yield_commitment": 0.0,
        "speed_reduction_conjunction": 0.0,
        "lateral_stability_conjunction": 0.0
    }

    # Check for perceptual claims about pedestrians
    if any(pc.entity == 'pedestrian' and pc.state == 'crossing' for pc in claims.perceptual):
        scores["saw_pedestrian"] = 0.1

    # Check for commitment to yield
    if any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments):
        scores["yield_commitment"] = 0.1

    # Check for speed reduction in the trajectory with timing
    if traj.n_waypoints > 0:
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if speed_drop >= 6.0 and 5.0 <= min_speed_time <= 6.4:  # Ensure timing is correct
            if any(cc.maneuver == 'yield' for cc in claims.commitments):
                scores["speed_reduction_conjunction"] = 0.5

    # Check for lateral stability with claims
    if abs(traj.final_lateral_offset_m) <= 0.25:
        if any(pc.entity == 'pedestrian' for pc in claims.perceptual):
            scores["lateral_stability_conjunction"] = 0.3

    return scores

def reward(claims, traj):
    """Reward function for scene with decisive event: Yield to pedestrian.
    Thresholds: speed drop >= 6.0 m/s with timing, final lateral offset <= 0.25 m."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
