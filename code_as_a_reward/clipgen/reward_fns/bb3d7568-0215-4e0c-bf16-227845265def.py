"""clip bb3d7568-0215-4e0c-bf16-227845265def - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for scene bb3d7568-0215-4e0c-bf16-227845265def:
    - Decisive Event 1: Pedestrian Crossing
      - Perceptual: 'pedestrian'
      - Commitment: 'decelerate' (speed_profile)
      - Trajectory: Speed drop >= 1.0 m/s, graded, with timing condition
    """
    perceptual_pedestrian = any(p.entity == 'pedestrian' for p in claims.perceptual)

    commitment_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory calculations
    speed_series = np.array(traj.speed_mps)

    # Speed drop calculation with timing condition
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 6.4))
    speed_drop = initial_speed - min_speed_after
    min_speed_time = 0.0 + np.argmin(window(speed_series, traj.dt_s, 0.0, 6.4)) * traj.dt_s

    # Component scores
    pedestrian_mention = 0.05 if perceptual_pedestrian else 0.0

    # Slowing component with timing condition
    slowing = 0.65 * min(1.0, speed_drop / 2.0) if commitment_decelerate and min_speed_time <= 2.0 else 0.0

    return {
        "pedestrian_mention": pedestrian_mention,
        "slowing": slowing
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
