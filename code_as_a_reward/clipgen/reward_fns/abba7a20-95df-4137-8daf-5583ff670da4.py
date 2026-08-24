"""clip abba7a20-95df-4137-8daf-5583ff670da4 - attempt 5/5 - gate PASS (pos 0.70, max pert 0.05, real rollout argmax 0)"""
def components(claims, traj):
    """Components for scene with pedestrian crossing and yielding intent.
    
    Decisive Event: Pedestrian crossing at a crosswalk.
    - Perceptual mention: pedestrian or crosswalk.
    - Commitment: Yielding behavior (speed_profile='decelerate').
    - Trajectory: Show a speed reduction of at least 4.0 m/s within the window.
    """
    # Initialize component scores
    perceptual_mention = 0.0
    commitment_yield = 0.0

    # Check for perceptual mention of pedestrian or crosswalk
    if any(p.entity in ('pedestrian', 'crosswalk') for p in claims.perceptual):
        perceptual_mention = 0.05  # Small weight for mention

    # Check for commitment to yield (speed_profile='decelerate')
    if any(c.speed_profile == 'decelerate' for c in claims.commitments):
        # Calculate speed reduction
        initial_speed = traj.speed_mps[0]
        min_speed_after = np.min(window(traj.speed_mps, traj.dt_s, 1.0, 6.4))
        speed_drop = initial_speed - min_speed_after

        # Graded speed reduction factor
        speed_reduction = min(1.0, speed_drop / 4.0)  # Floor at 4.0 m/s drop

        # Combine commitment and trajectory
        if speed_drop >= 4.0:
            commitment_yield = 0.65 * speed_reduction  # Increased weight for conjunction

    return {
        "perceptual_mention": perceptual_mention,
        "commitment_yield": commitment_yield,
    }

def reward(claims, traj):
    return min(1.0, max(0.0, sum(components(claims, traj).values())))
