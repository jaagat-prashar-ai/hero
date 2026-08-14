"""clip dea17a23-9d5c-412a-85a8-99e30146c434 - attempt 1/5 - gate PASS (pos 0.80, max pert 0.11, real rollout argmax 9)"""
def components(claims, traj):
    """Components for scoring the rollout based on the decisive event of decelerating to maintain a safe distance from a vehicle ahead.
    
    Decisive Event: Deceleration to Maintain Safe Distance from the Flatbed Truck Ahead
    - Perceptual mention of a vehicle ahead (entity family: vehicle_generic, lead_vehicle).
    - Commitment to decelerate (speed_profile='decelerate').
    - Trajectory should show a speed drop of at least 2.65 m/s within the window, graded above this floor.
    """

    # Initialize component scores
    perceptual_vehicle = 0.0
    decelerate_commitment = 0.0
    speed_drop_execution = 0.0

    # Check for perceptual mention of a vehicle
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        perceptual_vehicle = 0.1  # Small weight for mention

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop execution
        speed_drop_execution = 0.5 * min(1.0, speed_drop / 5.3)

        # Combine commitment and execution
        decelerate_commitment = 0.4 * (0.5 * min(1.0, speed_drop / 5.3))

    return {
        "perceptual_vehicle": perceptual_vehicle,
        "decelerate_commitment": decelerate_commitment,
        "speed_drop_execution": speed_drop_execution
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
