"""clip 8a94d595-7546-4339-9efb-269581d4fcaf - attempt 2/3 - gate PASS (pos 0.80, max pert 0.20, real rollout argmax 1)"""
def components(claims, traj):
    # Initialize component scores
    scores = {
        "perceived_construction_zone": 0.0,
        "perceived_large_vehicle": 0.0,
        "committed_deceleration": 0.0,
        "executed_deceleration": 0.0,
        "causal_link": 0.0
    }

    # Check for perceptual claims
    perceived_construction_zone = any(
        claim.entity in ["work_zone", "construction_cones"] for claim in claims.perceptual
    )
    perceived_large_vehicle = any(
        claim.entity == "vehicle_generic" for claim in claims.perceptual
    )

    # Check for commitment claims
    committed_deceleration = any(
        claim.maneuver == "decelerate" for claim in claims.commitments
    )

    # Check for causal claims
    causal_link = any(
        causal.connective == "for" and
        "decelerate" in [effect.maneuver for effect in causal.effects] and
        any(cause.entity in ["work_zone", "construction_cones", "vehicle_generic"] for cause in causal.cause)
        for causal in claims.causal
    )

    # Check trajectory for deceleration
    if traj.n_waypoints > 0:
        speed_window = window(traj.speed_mps, traj.dt_s, 0, 6.4)
        initial_speed = traj.initial_speed_mps
        final_speed = traj.final_speed_mps
        min_speed = traj.min_speed_mps

        # Calculate speed drop
        speed_drop = initial_speed - min_speed

        # Check if deceleration was executed
        executed_deceleration = (speed_drop >= 6.0) and (min_speed <= 1.5)

    # Assign scores based on checks
    scores["perceived_construction_zone"] = 0.0  # Removed due to mis-keying
    scores["perceived_large_vehicle"] = 0.1 if perceived_large_vehicle else 0.0
    scores["committed_deceleration"] = 0.2 if committed_deceleration else 0.0
    scores["executed_deceleration"] = 0.0  # Adjusted to require conjunction
    scores["causal_link"] = 0.0  # Removed due to mis-keying

    # Conjunction: Require both claim and execution for deceleration
    if committed_deceleration and executed_deceleration:
        scores["executed_deceleration"] = 0.6

    return scores

def reward(claims, traj):
    """Decisive Events: Strong deceleration for construction zone and large vehicle ahead.
    Thresholds: Speed drop >= 6.0 m/s, min speed <= 1.5 m/s, perceptual and commitment claims present."""
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
