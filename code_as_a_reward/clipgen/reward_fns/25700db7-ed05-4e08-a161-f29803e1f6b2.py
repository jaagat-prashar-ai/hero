"""clip 25700db7-ed05-4e08-a161-f29803e1f6b2 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with a pedestrian crossing at a crosswalk.
    Decisive event: yield to pedestrian (track 90) crossing the road.
    Thresholds: speed increase floor at 0.0 m/s, graded up to 5.0 m/s.
    Perceptual mention of 'pedestrian' or 'crosswalk' expected.
    """

    # Initialize component scores
    perceptual_pedestrian = 0.0
    yield_execution = 0.0

    # Check for perceptual mentions of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_pedestrian = 0.1

    # Check for commitment to yield (decelerate family)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate the speed increase over the trajectory
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_increase = final_speed - initial_speed

        # Graded factor for speed increase, floor at 0.0 m/s, graded up to 5.0 m/s
        yield_execution = 0.6 * min(1.0, speed_increase / 5.0)

    return {
        "perceptual_pedestrian": perceptual_pedestrian,
        "yield_execution": yield_execution,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
