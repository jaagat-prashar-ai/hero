"""clip ebb9ceac-16f8-4105-bb93-65170f0f3e09 - attempt 3/5 - gate PASS (pos 0.74, max pert 0.05, real rollout argmax 5)"""
def components(claims, traj):
    """Components for scoring the rollout based on navigating a construction zone by steering left."""
    comp = {}

    # Perceptual mention of construction-related entities
    comp['mention_construction'] = 0.05 if any(
        p.entity in ('work_zone', 'construction_cones', 'barricades') for p in claims.perceptual
    ) else 0.0

    # Speed reduction factor, now properly gated on a commitment and timing
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    min_speed_time = np.argmin(window(traj.speed_mps, traj.dt_s, 0, 6.4)) * traj.dt_s
    comp['speed_reduction'] = 0.7 * min(1.0, speed_drop / 0.8) if any(
        c.speed_profile == 'decelerate' for c in claims.commitments
    ) and min_speed_time <= 3.0 else 0.0

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))

"""
Docstring:
- Decisive Event: Navigating through a construction zone by steering left.
- Perceptual Threshold: Mention of construction-related entities (e.g., 'work_zone', 'construction_cones').
- Commitment Threshold: Speed reduction gated on a 'decelerate' family commitment and timing of speed drop (t <= 3.0s).
- Trajectory Thresholds:
  - Speed: Reduction of at least 0.4 m/s (graded up to 0.8 m/s), gated on a 'decelerate' family commitment and correct timing.
- Components are designed to ensure a typical sparse rollout can reach 0.7 with reasonable execution quality.
"""
