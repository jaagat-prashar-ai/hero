"""clip 5a8cca28-67c0-4c14-af65-76b875a026c4 - attempt 3/3 - gate PASS (pos 1.00, max pert 0.50, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring the rollout based on the decisive event of yielding to the pedestrian.
    Scene-derived thresholds:
    - Yielding involves reducing speed significantly (target minimum ~1.5-2.0 m/s) around t=4.5s.
    - Lateral offset should avoid the pedestrian, with a final offset around +0.94 m.
    - Perceptual claims should identify the pedestrian and crosswalk.
    - Commitment should include a yield maneuver with deceleration.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check perceptual claims
    saw_pedestrian = any(pc.entity == 'pedestrian' and pc.state == 'crossing' for pc in claims.perceptual)
    saw_crosswalk = any(pc.entity == 'crosswalk' and pc.state == 'crossing' for pc in claims.perceptual)

    if saw_pedestrian and saw_crosswalk:
        perceptual_score = 0.1

    # Check commitment claims
    committed_to_yield = any(cc.maneuver == 'yield' and cc.speed_profile == 'decelerate' for cc in claims.commitments)

    if committed_to_yield:
        commitment_score = 0.1

    # Check trajectory execution with timing and direction
    if traj.n_waypoints > 0:
        # Speed reduction check with timing and direction
        speed_window = window(traj.speed_mps, traj.dt_s, 4.0, 5.5)
        min_speed = np.min(speed_window) if len(speed_window) > 0 else float('inf')
        speed_reduction = traj.initial_speed_mps - min_speed

        # Ensure speed reduction occurs in the correct direction and time
        if speed_reduction >= 3.0 and min_speed <= 2.0 and traj.speed_mps[0] > traj.speed_mps[-1] and committed_to_yield:
            trajectory_score += 0.5

        # Lateral offset check
        final_lateral_offset = traj.final_lateral_offset_m
        if abs(final_lateral_offset) <= 1.0 and committed_to_yield:
            trajectory_score += 0.3

    return {
        "perceptual_score": perceptual_score,
        "commitment_score": commitment_score,
        "trajectory_score": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
