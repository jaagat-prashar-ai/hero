"""clip 642d2d7d-f00c-44af-a75b-21ccc10235e0 - attempt 1/5 - gate PASS (pos 0.95, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Maintain safe distance from lead vehicle: decelerate commitment and speed drop.
    - Yield to pedestrians: decelerate commitment and speed drop.
    - Perceptual mentions of vehicles and pedestrians.
    Scene-derived thresholds: speed drop >= 0.25 m/s for deceleration events.
    """

    # Initialize component scores
    maintain_distance_score = 0.0
    yield_to_pedestrians_score = 0.0
    vehicle_mention_score = 0.0
    pedestrian_mention_score = 0.0

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        vehicle_mention_score = 0.05  # Small weight for vehicle mention

    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        pedestrian_mention_score = 0.05  # Small weight for pedestrian mention

    # Check for commitment to maintain safe distance (decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for maintaining safe distance
        maintain_distance_score = 0.45 * min(1.0, speed_drop / 0.5)

    # Check for commitment to yield to pedestrians (decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for yielding to pedestrians
        yield_to_pedestrians_score = 0.45 * min(1.0, speed_drop / 0.5)

    # Return component scores
    return {
        "maintain_distance": maintain_distance_score,
        "yield_to_pedestrians": yield_to_pedestrians_score,
        "vehicle_mention": vehicle_mention_score,
        "pedestrian_mention": pedestrian_mention_score,
    }

def reward(claims, traj):
    # Calculate the total reward as the clamped sum of components
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
