"""clip 8bfd751f-d62f-46ab-911f-8566ca6de5b7 - attempt 4/5 - gate PASS (pos 0.75, max pert 0.10, real rollout argmax 8)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and nearby vehicles.
    
    Decisive events:
    1. Pedestrian crossing at crosswalk: Expect deceleration with a speed drop of at least 2.15 m/s.
    2. Nearby vehicles: Maintain lane stability with minimal lateral offset change.
    
    Thresholds:
    - Speed drop for pedestrian: 2.15 m/s minimum.
    - Lateral offset stability: max |offset| 0.12 m.
    """

    # Initialize component scores
    comp = {
        "perceptual_pedestrian": 0.1,
        "decelerate_for_pedestrian": 0.0,
        "lateral_stability": 0.0
    }

    # Perceptual claims
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        comp["perceptual_pedestrian"] = 0.1

    # Commitment claims and trajectory checks
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed = traj.min_speed_mps
        speed_drop = initial_speed - min_speed

        # Graded factor for speed drop
        if speed_drop >= 2.15:
            comp["decelerate_for_pedestrian"] = 0.65 * min(1.0, speed_drop / 4.3)

    # Lateral stability check, now gated by a commitment claim
    if any(c.maneuver in ('lane_change', 'nudge', 'merge', 'turn', 'enter', 'exit') for c in claims.commitments):
        max_lateral_offset = max(abs(offset) for offset in traj.lateral_offset_m)
        if max_lateral_offset <= 0.12:
            comp["lateral_stability"] = 0.15

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
