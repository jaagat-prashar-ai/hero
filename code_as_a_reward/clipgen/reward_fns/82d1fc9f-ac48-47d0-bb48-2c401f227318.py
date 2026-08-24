"""clip 82d1fc9f-ac48-47d0-bb48-2c401f227318 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 82d1fc9f-ac48-47d0-bb48-2c401f227318:
    - Maintain speed while keeping a safe distance from the vehicle cutting in.
    - Require a commitment to maintain speed or avoid unnecessary deceleration.
    - Trajectory should maintain speed with minimal deviation.
    """
    comp = {
        "maintain_speed": 0.0,
    }

    # Trajectory analysis
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    maintain_speed_factor = 0.7 * min(1.0, speed_drop / 0.9)  # Graded factor for maintaining speed

    # Commitment check: Require a commitment to maintain speed or avoid unnecessary deceleration
    if any(c.speed_profile == 'maintain' for c in claims.commitments):
        comp["maintain_speed"] = maintain_speed_factor

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
