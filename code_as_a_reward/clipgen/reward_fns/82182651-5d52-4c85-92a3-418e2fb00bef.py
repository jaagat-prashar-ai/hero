"""clip 82182651-5d52-4c85-92a3-418e2fb00bef - attempt 2/5 - gate PASS (pos 1.00, max pert 0.08, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scoring a rollout based on yielding to pedestrians at a crosswalk.
    
    Decisive Event: Yield to pedestrians crossing the road at the crosswalk.
    - Perceptual: Mention of 'pedestrian' or 'crosswalk'.
    - Commitment: Deceleration (speed_profile='decelerate').
    - Trajectory: Speed reduction of at least 0.4 m/s, graded with a floor at half the GT drop.
    """
    perceptual_weight = 0.05
    commitment_weight = 0.65
    trajectory_weight = 0.30

    # Perceptual component: mention of pedestrian or crosswalk
    saw_pedestrian = any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual)
    perceptual_score = perceptual_weight if saw_pedestrian else 0.0

    # Commitment component: decelerate (yield, stop, wait, decelerate)
    has_decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory component: speed reduction
    speed_series = np.array(traj.speed_mps)
    initial_speed = traj.initial_speed_mps
    min_speed_after = np.min(window(speed_series, traj.dt_s, 0, 6.4))
    speed_drop = initial_speed - min_speed_after
    trajectory_score = 0.0

    if has_decelerate_commitment:
        # Graded trajectory score based on speed drop
        trajectory_score = trajectory_weight * min(1.0, speed_drop / 1.3)  # Adjusted for realistic drop

    # Combine components
    components = {
        "perceptual_mention": perceptual_score,
        "decelerate_commitment": commitment_weight if has_decelerate_commitment and speed_drop >= 0.4 else 0.0,
        "trajectory_execution": trajectory_score
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
