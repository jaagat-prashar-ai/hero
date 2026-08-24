"""clip cca7366b-b9ef-45a9-8a6d-68a844b28330 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.30, real rollout argmax 6)"""
def components(claims, traj):
    """Components for reward function based on decisive events:
    - Deceleration for lead vehicle with trailer: expect 'decelerate' commitment and speed drop.
    - Perceptual mention of 'lead_vehicle' or similar.
    - Trajectory speed drop graded from 1.0 m/s floor.
    """

    # Initialize component scores
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "commitment_decelerate": 0.0,
        "trajectory_deceleration": 0.0,
    }

    # Perceptual mention of lead vehicle or similar
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.1

    # Commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        comp["commitment_decelerate"] = 0.2

        # Trajectory speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s))
        speed_drop = initial_speed - min_speed_after
        comp["trajectory_deceleration"] = 0.5 * min(1.0, speed_drop / 2.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
