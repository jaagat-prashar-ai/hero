"""clip 002d7967-a4eb-4d6c-a71d-a956e058cda5 - attempt 3/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 3)"""
def components(claims, traj):
    """Components for scene 002d7967-a4eb-4d6c-a71d-a956e058cda5:
    - Deceleration to yield to traffic: speed drop >= 0.75 m/s, claim family 'decelerate'
    - Perceptual mention of nearby traffic: entity family 'vehicle_generic'
    """
    comp = {
        "decelerate_execution": 0.0,
        "perceptual_mention": 0.0,
    }

    # Check for perceptual mention of nearby traffic
    if any(p.entity in ('vehicle_generic', 'cross_traffic') for p in claims.perceptual):
        comp["perceptual_mention"] = 0.1

    # Check for deceleration commitment and execution
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        # Ensure the minimum speed occurs at a reasonable time (before t=3.5s)
        min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
        if min_speed_time <= 3.5:
            # Graded factor for deceleration execution
            comp["decelerate_execution"] = 0.6 * min(1.0, speed_drop / 1.5)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
