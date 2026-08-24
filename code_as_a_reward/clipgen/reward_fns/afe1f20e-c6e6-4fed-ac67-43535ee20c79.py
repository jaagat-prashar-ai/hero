"""clip afe1f20e-c6e6-4fed-ac67-43535ee20c79 - attempt 2/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 11)"""
def components(claims, traj):
    """Components for scene with deceleration to stop due to lead vehicle presence.
    
    Decisive Events:
    1. Deceleration to Stop: Expect a deceleration commitment and a trajectory
       showing a speed drop of at least 3.0 m/s by t=4.4 s.
    2. Presence of Multiple Vehicles: Expect mention of vehicles and a similar
       deceleration trajectory.
    
    Scene-derived thresholds:
    - Speed drop floor: 2.0 m/s (half of the positive's 4.0 m/s drop)
    - Timing: Deceleration primarily within the first 4.4 seconds
    """
    # Initialize component scores
    comp = {
        "perceptual_vehicle": 0.05,  # Reduced weight for mention-only
        "decelerate_commitment_execution": 0.0
    }

    # Perceptual check: mention of vehicles
    if any(p.entity in ('lead_vehicle', 'vehicle_generic', 'stopped_vehicle') for p in claims.perceptual):
        comp["perceptual_vehicle"] = 0.05

    # Commitment and trajectory check: deceleration
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        speed_series = np.array(traj.speed_mps)
        initial_speed = traj.initial_speed_mps
        min_speed_after = np.min(window(speed_series, traj.dt_s, 0.0, 4.4))
        speed_drop = initial_speed - min_speed_after

        # Graded factor for speed drop
        if speed_drop >= 2.0:  # Floor at half the positive's drop
            comp["decelerate_commitment_execution"] = 0.65 * min(1.0, speed_drop / 4.0)

    return comp

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
