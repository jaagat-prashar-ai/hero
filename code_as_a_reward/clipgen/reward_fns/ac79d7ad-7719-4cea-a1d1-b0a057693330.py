"""clip ac79d7ad-7719-4cea-a1d1-b0a057693330 - attempt 1/5 - gate PASS (pos 0.93, max pert 0.53, real rollout argmax 11)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive Event: Maintain a safe distance from the lead vehicle while following traffic cones.
    - Perceptual: Mention of 'lead_vehicle' or 'construction_cones'
    - Commitment: Maintain speed (absence of 'decelerate')
    - Trajectory: Maintain speed close to initial speed (13.0 m/s)
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('lead_vehicle', 'construction_cones') for p in claims.perceptual):
        perceptual_score = 0.1  # Small weight for mention

    # Check for commitment to maintain speed (absence of 'decelerate')
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed maintenance factor
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_maintenance_factor = max(0.0, min(1.0, (final_speed - initial_speed) / 6.0))
        commitment_score = 0.4 * speed_maintenance_factor

    # Trajectory execution: Maintain speed close to initial speed
    speed_window = window(traj.speed_mps, traj.dt_s, 0.0, traj.n_waypoints * traj.dt_s)
    min_speed_after = np.min(speed_window)
    if min_speed_after >= 12.0:  # Floor at half the scene's magnitude
        trajectory_score = 0.5 * min(1.0, (min_speed_after - 12.0) / 6.0)

    return {
        "perceptual_mention": perceptual_score,
        "commitment_maintain_speed": commitment_score,
        "trajectory_maintain_speed": trajectory_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
