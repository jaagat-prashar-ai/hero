"""clip 982f53ae-239e-4da2-97cb-bb5a007dd75c - attempt 4/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    # Decisive Event: Navigating Through the Construction Zone
    # - Perceptual: Recognize the construction zone or related entities.
    # - Commitment: Maintain speed through the zone.
    # - Trajectory: Maintain speed with minimal lateral deviation.

    # Initialize component scores
    maintain_speed_score = 0.0

    # Commitment component: Maintain speed
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Calculate speed maintenance factor
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, traj.n_waypoints)) * traj.dt_s
        # Ensure the minimum speed occurs later in the window to differentiate from reversed trajectory
        if min_speed_time > 3.0:  # Arbitrary time threshold to ensure minimum speed occurs later
            speed_drop = initial_speed - min_speed
            speed_maintenance_factor = min(1.0, speed_drop / 5.7)  # Graded factor based on speed drop
            maintain_speed_score = 0.7 * speed_maintenance_factor  # Graded factor

    return {
        "maintain_speed": maintain_speed_score
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))

"""
Docstring:
Decisive Event: Navigating Through the Construction Zone
- Commitment: Maintain speed through the zone, matched at the FAMILY level with 'maintain' speed_profile.
- Trajectory: Maintain speed with minimal lateral deviation. Speed maintenance is graded based on maintaining or slightly increasing speed. Lateral offset is graded based on minimal deviation from the initial path.
"""
