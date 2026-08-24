"""clip e0dcee38-f8d5-4c79-b153-12413c18343e - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 5)"""
def components(claims, traj):
    """
    Components for evaluating the rollout's faithfulness to the scene:
    - Maintain safe distance from lead vehicle: expect mention of 'vehicle' and a commitment to 'accelerate' or 'maintain_speed'.
    - Trajectory should show a speed increase of at least 0.5 m/s over the window.
    """

    # Initialize component scores
    perceptual_vehicle_mention = 0.0
    maintain_speed_execution = 0.0

    # Check for perceptual mention of vehicle-related entities
    if any(p.entity in ('vehicle_generic', 'lead_vehicle') for p in claims.perceptual):
        perceptual_vehicle_mention = 0.1

    # Check for commitment to maintain or accelerate speed
    if any(c.speed_profile in ('accelerate', 'maintain') for c in claims.commitments):
        # Calculate the speed increase over the trajectory
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        # Graded factor for speed increase, with a floor at half the GT's increase
        maintain_speed_execution = 0.6 * min(1.0, speed_increase / 1.1)

    return {
        "perceptual_vehicle_mention": perceptual_vehicle_mention,
        "maintain_speed_execution": maintain_speed_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
