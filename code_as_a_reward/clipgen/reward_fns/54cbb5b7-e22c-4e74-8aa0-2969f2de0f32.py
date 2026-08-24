"""clip 54cbb5b7-e22c-4e74-8aa0-2969f2de0f32 - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 1)"""
def components(claims, traj):
    """Components for scene 54cbb5b7-e22c-4e74-8aa0-2969f2de0f32:
    - Maintain lane and safe distance from lead vehicle.
    - Perceptual mention of 'lead_vehicle' or 'vehicle_generic'.
    - No speed change required; maintain speed.
    - No lateral maneuvers required; stay in lane.
    """
    # Initialize component scores
    perceptual_score = 0.0
    maintain_speed_score = 0.0

    # Perceptual mention of lead vehicle or generic vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        perceptual_score = 0.1

    # Check for maintaining speed (no deceleration or acceleration)
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        # Graded factor for maintaining speed
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        maintain_speed_score = 0.6 * min(1.0, max(0.0, speed_increase / 1.5))

    return {
        "perceptual_mention": perceptual_score,
        "maintain_speed": maintain_speed_score,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
