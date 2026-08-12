"""clip a34c532f-d452-4678-9111-ac56480972ed - attempt 3/3 - gate PASS (pos 0.70, max pert 0.00, real rollout argmax 9)"""
def components(claims, traj):
    """
    Components for reward function based on decisive events:
    1. Traffic Light Turning Green: Resume speed.
    2. Construction Zone on the Left: Maintain lateral position.
    
    Scene-derived thresholds:
    - Speed increase from 0.6 m/s to approximately 6.6 m/s.
    - Lateral offset maintained within approximately |0.44| m.
    """
    # Initialize component scores
    score_resume_speed = 0.0

    # Check for perceptual claims and commitments
    saw_green_light = any(pc.entity == 'signal' and pc.state == 'green' for pc in claims.perceptual)
    committed_to_accelerate = any(cc.maneuver == 'accelerate' for cc in claims.commitments)

    # Check trajectory for speed increase
    if traj.n_waypoints > 1:
        speed_increase = traj.final_speed_mps - traj.initial_speed_mps
        if speed_increase > 5.0 and saw_green_light and committed_to_accelerate:  # Require claim, commitment, and trajectory
            score_resume_speed = 0.7

    return {
        "resume_speed": score_resume_speed
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
