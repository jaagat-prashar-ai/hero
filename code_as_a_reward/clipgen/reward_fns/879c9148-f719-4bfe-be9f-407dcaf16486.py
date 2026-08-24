"""clip 879c9148-f719-4bfe-be9f-407dcaf16486 - attempt 2/5 - gate PASS (pos 0.83, max pert 0.00, real rollout argmax 7)"""
def components(claims, traj):
    """
    Components for scene 879c9148-f719-4bfe-be9f-407dcaf16486:
    - Creeping forward: expect 'decelerate' commitment with gradual speed increase.
    - Perceptual mentions: expect 'vehicle_generic' or similar entity mention.
    Scene-derived thresholds: speed increase up to 2.6 m/s, with graded factors.
    """

    # Initialize component scores
    comp = {
        "perceptual_mention": 0.0,
        "creeping_forward": 0.0,
    }

    # Check for perceptual mentions
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Check for 'decelerate' commitment
    has_decelerate_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)

    # Trajectory analysis
    if traj.n_waypoints > 0:
        # Creeping forward (0.0 - 6.4 s)
        final_speed = traj.final_speed_mps
        if has_decelerate_commitment:
            speed_increase = final_speed - traj.initial_speed_mps
            comp["creeping_forward"] = 0.9 * min(1.0, speed_increase / 2.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
