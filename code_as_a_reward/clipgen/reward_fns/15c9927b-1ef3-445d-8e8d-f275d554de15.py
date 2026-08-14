"""clip 15c9927b-1ef3-445d-8e8d-f275d554de15 - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene 15c9927b-1ef3-445d-8e8d-f275d554de15:
    - Navigate through a construction zone with traffic cones on the left side.
    - Maintain lane position with minimal speed change.
    - Commitment: maintain speed with minimal change.
    - Trajectory: Maintain speed change within ±1.0 m/s, with time-sensitive condition.
    """
    comp = {}

    # Commitment component: maintain speed with minimal change
    speed_commitment = any(
        c.speed_profile in ('maintain', 'decelerate') for c in claims.commitments
    )
    # Check if the minimum speed occurs at the start of the window
    min_speed_idx = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints * traj.dt_s))
    min_speed_time = min_speed_idx * traj.dt_s
    speed_change = traj.final_speed_mps - traj.initial_speed_mps
    comp['maintain_speed'] = 0.70 * speed_commitment * (abs(speed_change) <= 1.0) * (min_speed_time < 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
