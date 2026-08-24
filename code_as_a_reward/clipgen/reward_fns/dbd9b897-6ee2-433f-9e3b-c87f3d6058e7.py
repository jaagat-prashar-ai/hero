"""clip dbd9b897-6ee2-433f-9e3b-c87f3d6058e7 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.26, real rollout argmax 4)"""
def components(claims, traj):
    """Components for scene dbd9b897-6ee2-433f-9e3b-c87f3d6058e7:
    - Deceleration to maintain safe distance from pedestrian (Track 66)
      - Perceptual: 'pedestrian'
      - Commitment: 'decelerate' (speed_profile)
      - Trajectory: Speed drop >= 0.3 m/s, graded
    - Automobiles on the left (Tracks 132 and 68)
      - Perceptual: 'vehicle_generic'
      - Commitment: 'decelerate' (speed_profile)
      - Trajectory: Speed drop >= 0.3 m/s, graded
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_vehicle": 0.0,
        "decelerate_execution": 0.0,
    }

    # Perceptual components
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    if any(p.entity == 'vehicle_generic' for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Commitment and trajectory components
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded trajectory factor for deceleration
        comp["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 0.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
