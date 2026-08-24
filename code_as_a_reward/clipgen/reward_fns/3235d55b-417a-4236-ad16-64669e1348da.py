"""clip 3235d55b-417a-4236-ad16-64669e1348da - attempt 1/5 - gate PASS (pos 0.70, max pert 0.10, real rollout argmax 11)"""
def components(claims, traj):
    """Components for maintaining speed while following the lead vehicle.
    
    Decisive Event: Maintain speed while following the lead vehicle.
    - Perceptual: Mention of lead vehicle.
    - Commitment: Maintain or slightly increase speed.
    - Trajectory: Speed should remain constant or slightly increase.
    """
    comp = {
        "perceptual_lead_vehicle": 0.0,
        "maintain_speed": 0.0,
    }

    # Perceptual component: Mention of lead vehicle
    if any(p.entity in ('lead_vehicle', 'vehicle_generic') for p in claims.perceptual):
        comp["perceptual_lead_vehicle"] = 0.1

    # Commitment component: Maintain or slightly increase speed
    if any(c.speed_profile in ('maintain', 'accelerate') for c in claims.commitments):
        # Calculate the speed increase factor
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        speed_increase = final_speed - initial_speed
        # Graded factor for speed increase, floor at half the GT's increase (0.3 m/s)
        comp["maintain_speed"] = 0.6 * min(1.0, (speed_increase / 0.3))

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
