"""clip c388b989-903e-4eb5-835b-cc0456d0b6bc - attempt 1/5 - gate PASS (pos 0.92, max pert 0.35, real rollout argmax 6)"""
def components(claims, traj):
    """
    Components for scoring a rollout based on the scene's decisive events:
    1. Stop and wait for pedestrian crossing: Expect a 'decelerate' commitment and a speed drop of at least 1.0 m/s by t=3.7 s.
    2. Maintain position relative to riders: Expect minimal lateral movement, with max |offset| around 0.13 m.
    Perceptual mentions are small additive scores, and trajectory factors are graded above generous floors.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.0,
        "perceptual_rider": 0.0,
        "stop_executed": 0.0,
        "maintain_position": 0.0
    }

    # Perceptual claims
    if any(p.entity in ('pedestrian',) for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.05

    if any(p.entity in ('cyclist',) for p in claims.perceptual):
        comp["perceptual_rider"] = 0.05

    # Commitment and trajectory for stopping
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Graded factor for speed drop
        comp["stop_executed"] = 0.6 * min(1.0, speed_drop / 2.0)

    # Trajectory for maintaining position
    lateral_offset = max(abs(traj.final_lateral_offset_m), abs(traj.lateral_offset_m[0]))
    comp["maintain_position"] = 0.3 * min(1.0, (0.26 - lateral_offset) / 0.26)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
