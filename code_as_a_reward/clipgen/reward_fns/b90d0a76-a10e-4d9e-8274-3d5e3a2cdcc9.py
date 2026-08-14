"""clip b90d0a76-a10e-4d9e-8274-3d5e3a2cdcc9 - attempt 2/5 - gate PASS (pos 0.90, max pert 0.34, real rollout argmax 0)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on its reasoning and trajectory.
    Decisive events:
    1. Protruding Object (Track 237) and Automobile (Track 229): Require deceleration.
    Scene-derived thresholds:
    - Speed drop of at least 0.8 m/s for deceleration.
    - Perceptual mention of relevant entities.
    """

    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.0,
        "decelerate_execution": 0.0,
    }

    # Check for perceptual claims
    if any(p.entity in ('vehicle_generic', 'lead_vehicle', 'stopped_vehicle', 'cutin_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.1

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.speed_mps[0]
        min_speed_after = min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded deceleration execution score
        comp["decelerate_execution"] = 0.9 * min(1.0, speed_drop / 1.6)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
