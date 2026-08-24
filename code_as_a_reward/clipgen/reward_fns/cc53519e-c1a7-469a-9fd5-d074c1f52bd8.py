"""clip cc53519e-c1a7-469a-9fd5-d074c1f52bd8 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 7)"""
def components(claims, traj):
    """Components for evaluating the rollout's faithfulness to the scene.
    
    Decisive Events:
    1. Stopping for the lead vehicle at the green traffic light.
       - Perceptual entity: 'lead_vehicle', 'signal', 'vehicle_generic'
       - Commitment family: 'decelerate' (stop/yield/wait/decelerate)
       - Trajectory: Speed drop of at least 1.75 m/s by t=3.5s, graded.
    """
    perceptual_weight = 0.05
    commitment_weight = 0.65

    # Perceptual component: Mention of relevant entities
    perceptual_entities = {'lead_vehicle', 'signal', 'vehicle_generic'}
    saw_lead_vehicle = any(p.entity in perceptual_entities for p in claims.perceptual)

    # Commitment component: Deceleration
    slowing_commitment = any(c.speed_profile == 'decelerate' for c in claims.commitments)
    speed_drop = traj.initial_speed_mps - traj.min_speed_mps
    speed_drop_factor = 0.65 * min(1.0, speed_drop / 1.75) if slowing_commitment else 0.0

    return {
        "saw_lead_vehicle": perceptual_weight if saw_lead_vehicle else 0.0,
        "decelerate_executed": speed_drop_factor,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
