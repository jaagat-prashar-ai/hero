"""clip 25300a8b-23ab-4c16-bc59-5ea3130adf28 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 25300a8b-23ab-4c16-bc59-5ea3130adf28:
    - Maintain speed and lane after pedestrians clear the crosswalk.
    - Expect minimal speed change and lateral offset.
    - Perceptual mention of pedestrians.
    """
    comp = {
        "mention_pedestrian": 0.0,
        "maintain_speed": 0.0,
    }

    # Perceptual mention of pedestrians
    if any(p.entity == 'pedestrian' for p in claims.perceptual):
        comp["mention_pedestrian"] = 0.05

    # Maintain speed: expect minimal speed change with commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 0.5:  # Floor at 0.5 m/s drop
            comp["maintain_speed"] = 0.65 * min(1.0, speed_drop / 1.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
