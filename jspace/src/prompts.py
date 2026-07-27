"""Fitting/eval prompt corpus for the Alpamayo J-lens.

The paper fits J_l on a pretraining-like distribution (~1000 prompts x 128
tokens; ~100 prompts is usable). For a driving VLA we bias the corpus toward
driving reasoning: built-in CoT-style prompts covering the 9 OOD event
clusters in nvidia/PhysicalAI-Autonomous-Vehicles, optionally topped up with
real chain-of-causation ("coc") strings from reasoning/ood_reasoning.parquet.
"""

from pathlib import Path

# Two hand-written reasoning prompts per OOD event_cluster, plus generic
# driving narration. Written in the declarative CoT style Alpamayo emits.
DRIVING_PROMPTS = [
    # WORK_ZONES_TEMP_TRAFFIC_CONTROL
    "Orange cones taper the right lane into the left ahead of a work zone. A flagger is holding a slow sign, so the ego vehicle should merge left early and reduce speed through the taper.",
    "A temporary traffic light controls alternating one-lane flow past road works. It is red for our direction, so the ego vehicle must stop at the stop line and wait for the oncoming platoon to clear.",
    # PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY
    "A group of pedestrians is clustered at the crosswalk and one has already stepped off the curb. The ego vehicle should brake smoothly now and yield until the crosswalk is fully clear.",
    "Pedestrians are walking along the shoulder close to the travel lane at night. The ego vehicle should shift laterally away from them and pass at reduced speed.",
    # SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR
    "The delivery van two cars ahead is reversing against traffic to reach a driveway. The ego vehicle should hold back, leave a large gap, and be ready to stop until its intent is clear.",
    "An oversize load escorted by pilot cars straddles both lanes ahead. Overtaking is unsafe, so the ego vehicle should follow at a distance and match the convoy's speed.",
    # CYCLISTS_AND_MICROMOBILITY_COMPLEX
    "A cyclist ahead is signaling a left turn while a scooter filters between lanes on the right. The ego vehicle should slow, let the cyclist commit to the turn, and avoid overtaking mid-intersection.",
    "A child on a bike wobbles at the edge of the bike lane. Give at least a full lane of clearance; if oncoming traffic prevents that, slow and follow until it is safe to pass.",
    # COMPLEX_INTERSECTION_INTERACTION
    "At a four-way stop, we arrived second but the first driver is waving us through. Proceed cautiously only after confirming cross traffic is actually yielding, then clear the intersection promptly.",
    "The traffic light turned green, but a car is still clearing the intersection from the left and a pedestrian is finishing the crossing. Delay the start until the box is empty.",
    # OTHER_LONGTAIL
    "Thick fog has cut visibility to about fifty meters. The ego vehicle should drop speed, increase following distance, and rely on lane markings rather than taillights ahead.",
    "A mattress is sliding off the pickup truck ahead onto the highway. Brake early, signal, and change lanes away from the falling cargo without swerving abruptly.",
    # EMERGENCY_INCIDENT_SCENE
    "An ambulance with lights and siren is approaching from behind in heavy traffic. The ego vehicle should signal, pull to the right edge, and stop until it has passed.",
    "Flares and a police car block the right lane at a crash scene. Merge left when safe, pass the scene slowly, and watch for responders stepping into the open lane.",
    # ROAD_DEBRIS_OR_SAFETY_TRACES
    "A shredded truck tire lies across the middle of our lane. Check the mirror, then steer around it within the lane if possible; otherwise brake and wait for a gap.",
    "Fresh skid marks and scattered glass suggest a recent collision beyond the crest. Slow before the crest and be prepared for stopped vehicles just out of sight.",
    # ANIMALS_BIRDS_ROADKILL
    "A deer is standing on the shoulder facing the road at dusk. Deer often bolt across; the ego vehicle should slow now and cover the brake until well past it.",
    "A flock of geese is crossing the road single file. Stop and wait for them to clear rather than weaving between the birds.",
    # Generic driving narration
    "Traffic on the highway is compressing ahead; brake lights ripple back toward us. Ease off the accelerator early to absorb the wave without a hard stop.",
    "The navigation route requires a right turn in two hundred meters and we are in the left lane. Signal, check the blind spot, and merge right behind the silver sedan.",
    "Rain has started and the road surface is glossy. Increase following distance, avoid standing water near the gutter, and brake earlier than usual.",
    "The lead vehicle's behavior is erratic, drifting across the lane line twice in the last minute. Increase the gap and pass only with ample clearance.",
]


def load_coc_prompts(parquet_path: str | Path, max_prompts: int = 1000) -> list[str]:
    """Extract chain-of-causation strings from ood_reasoning.parquet.

    One row per OOD clip; each row's `events` is a list of dicts whose `coc`
    field is the natural-language chain of causation. Rows/events without a
    non-empty coc are skipped. Stops as soon as max_prompts are collected.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["events"])
    prompts: list[str] = []
    for events in df["events"]:
        for event in events:
            coc = (event.get("coc") or "").strip()
            if coc:
                prompts.append(coc)
                if len(prompts) >= max_prompts:
                    return prompts
    return prompts


def corpus(parquet_path: str | Path | None = None, max_prompts: int = 1000) -> list[str]:
    """Built-in prompts, topped up with real coc strings when available."""
    prompts = list(DRIVING_PROMPTS)
    if parquet_path is not None:
        prompts += load_coc_prompts(parquet_path, max_prompts=max_prompts - len(prompts))
    seen: set[str] = set()
    return [p for p in prompts if not (p in seen or seen.add(p))][:max_prompts]
