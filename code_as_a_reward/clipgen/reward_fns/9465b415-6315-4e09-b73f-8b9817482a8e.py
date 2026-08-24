"""clip 9465b415-6315-4e09-b73f-8b9817482a8e - attempt 3/5 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene 9465b415-6315-4e09-b73f-8b9817482a8e:
    - Yield to the yield sign at the intersection: expect a 'decelerate' commitment
      and a speed reduction of at least 4.35 m/s (half of 8.7 m/s) within the window.
    """
    # Initialize component scores
    comp = {
        "decelerate_execution": 0.0
    }

    # Commitment to decelerate (yield/stop/wait/decelerate)
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed drop within the entire window
        speed_drop = traj.initial_speed_mps - traj.min_speed_mps
        if speed_drop >= 4.35:
            comp["decelerate_execution"] = 0.7 * min(1.0, speed_drop / 8.7)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
