"""clip 056535bc-814d-4b4f-834f-33848248104b - attempt 2/5 - gate PASS (pos 0.70, max pert 0.03, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with gentle deceleration to yield to traffic on the through road.
    
    Decisive Event:
    - Deceleration to yield to traffic on the through road.
    
    Scene-Derived Thresholds:
    - Speed drop: at least 1.35 m/s (half of the expert's 2.7 m/s drop).
    - Timing: Deceleration should occur primarily in the first half of the window.
    """

    # Initialize component scores
    deceleration_credit = 0.0

    # Check for deceleration commitment
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop
        initial_speed = traj.initial_speed_mps
        min_speed_after = traj.min_speed_mps
        speed_drop = initial_speed - min_speed_after

        # Graded deceleration factor
        deceleration_credit = 0.7 * min(1.0, speed_drop / 2.7)

    return {
        "deceleration_executed": deceleration_credit
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
