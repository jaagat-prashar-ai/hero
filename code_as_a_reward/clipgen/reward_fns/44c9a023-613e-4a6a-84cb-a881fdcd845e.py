"""clip 44c9a023-613e-4a6a-84cb-a881fdcd845e - attempt 1/5 - gate PASS (pos 1.00, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for navigating through a construction zone with slight speed adjustment."""
    comp = {
        "perceptual_construction": 0.0,
        "speed_deceleration": 0.0,
    }

    # Perceptual component: mention of construction-related entities
    if any(p.entity in ('work_zone', 'construction_cones', 'barricades', 'workers') for p in claims.perceptual):
        comp["perceptual_construction"] = 0.1

    # Commitment component: speed adjustment (decelerate family)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed drop during the window
        initial_speed = traj.initial_speed_mps
        min_speed = np.min(window(traj.speed_mps, traj.dt_s, 0, 6.4))
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop, floored at half the GT drop
        comp["speed_deceleration"] = 0.9 * min(1.0, speed_drop / 0.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
