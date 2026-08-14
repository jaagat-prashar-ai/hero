"""clip c0a5277b-8460-44fd-9265-1b0e68141d3e - attempt 4/5 - gate PASS (pos 0.81, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for the scene where the expert gently accelerates to pass a slow vehicle ahead.
    
    Decisive Event: Gentle acceleration to pass the slow vehicle ahead.
    - Perceptual mention: vehicle_generic
    - Commitment: accelerate (speed_profile='accelerate')
    - Trajectory: Speed increase from 23.9 m/s to 28.4 m/s, with a graded factor for speed gain.
    """

    # Initialize component scores
    accelerate_execution = 0.0

    # Check for acceleration commitment
    if any(c.speed_profile == 'accelerate' for c in claims.commitments):
        # Calculate speed increase
        speed_gain = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for speed increase, with a floor at half the GT increase
        speed_increase = 0.9 * min(1.0, speed_gain / 2.25)
        accelerate_execution = 0.9 * speed_increase  # Weight for having the commitment and execution

    # Return component scores
    return {
        "accelerate_execution": accelerate_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
