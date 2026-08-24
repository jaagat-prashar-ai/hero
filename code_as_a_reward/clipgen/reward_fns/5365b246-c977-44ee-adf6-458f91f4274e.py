"""clip 5365b246-c977-44ee-adf6-458f91f4274e - attempt 1/5 - gate PASS (pos 0.91, max pert 0.44, real rollout argmax 1)"""
def components(claims, traj):
    """
    Components for scene 5365b246-c977-44ee-adf6-458f91f4274e:
    - Decisive event: Pedestrian crossing at crosswalk.
    - Perceptual mention: 'pedestrian' or 'crosswalk'.
    - Commitment: 'decelerate' family (stop/yield/wait/decelerate).
    - Trajectory: Minor speed drop (floor 0.5 m/s), lateral offset increase (floor +3.0 m).
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    commitment_slowing = 0.0
    trajectory_slowing = 0.0
    trajectory_lateral = 0.0

    # Check for perceptual mentions of pedestrians or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to decelerate
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        trajectory_slowing = 0.5 * min(1.0, speed_drop / 1.0)  # Floor at 0.5 m/s drop

        # Calculate lateral offset increase
        lateral_offset_increase = traj.final_lateral_offset_m - 0.0  # Assuming initial offset is 0
        trajectory_lateral = 0.4 * min(1.0, lateral_offset_increase / 3.0)  # Floor at +3.0 m

        # Combine commitment and trajectory for slowing
        commitment_slowing = 0.3  # Base score for having the commitment

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "commitment_slowing": commitment_slowing,
        "trajectory_slowing": trajectory_slowing,
        "trajectory_lateral": trajectory_lateral
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
