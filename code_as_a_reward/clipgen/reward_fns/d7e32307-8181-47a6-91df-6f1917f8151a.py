"""clip d7e32307-8181-47a6-91df-6f1917f8151a - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the decisive event of stopping at a traffic light.
    Scene-derived thresholds:
    - Maintain speed close to 0.0 m/s (stopped position).
    - Maintain lateral offset close to 0.00 m (no lateral movement).
    - Perceptual mention of 'lead_vehicle' or 'signal'.
    """

    # Initialize component scores
    perceptual_score = 0.0
    commitment_score = 0.0
    trajectory_speed_score = 0.0
    trajectory_lateral_score = 0.0

    # Perceptual mention of relevant entities
    if any(p.entity in ('lead_vehicle', 'signal') for p in claims.perceptual):
        perceptual_score = 0.1

    # Commitment to decelerate (stop, yield, wait, decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Trajectory speed: maintain speed close to 0.0 m/s
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        trajectory_speed_score = 0.5 * min(1.0, max(0.0, 1.0 - abs(traj.min_speed_mps - 0.0) / 0.5))

        # Trajectory lateral: maintain lateral offset close to 0.00 m
        trajectory_lateral_score = 0.3 * min(1.0, max(0.0, 1.0 - abs(traj.final_lateral_offset_m - 0.0) / 0.5))

        # Commitment score is awarded if the commitment is present
        commitment_score = 0.1

    return {
        "perceptual_mention": perceptual_score,
        "commitment_execution": commitment_score,
        "trajectory_speed": trajectory_speed_score,
        "trajectory_lateral": trajectory_lateral_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
