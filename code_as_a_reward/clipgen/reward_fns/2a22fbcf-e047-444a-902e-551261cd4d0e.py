"""clip 2a22fbcf-e047-444a-902e-551261cd4d0e - attempt 3/5 - gate PASS (pos 0.78, max pert 0.00, real rollout argmax 6)"""
def components(claims, traj):
    """Components for scene 2a22fbcf-e047-444a-902e-551261cd4d0e:
    - Decisive event: Stopping behind a vehicle and workers.
    - Commitment: speed_profile='decelerate' (stop/yield/wait/decelerate).
    - Trajectory: Speed drop of at least 0.05 m/s by t=6.3 s.
    """
    commitment_decelerate = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Calculate speed drop
    initial_speed = traj.initial_speed_mps
    min_speed_after = traj.min_speed_mps
    speed_drop = initial_speed - min_speed_after

    # Graded trajectory factor for deceleration
    deceleration_factor = 0.8 * min(1.0, speed_drop / 0.05) if commitment_decelerate else 0.0

    components = {
        "decelerate_execution": deceleration_factor,
    }

    return components

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
