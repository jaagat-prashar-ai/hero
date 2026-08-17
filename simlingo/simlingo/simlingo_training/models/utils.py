
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch import Tensor
from simlingo_training.utils.custom_types import TrainingOutput


def intra_scene_contrastive_loss(
    z_text: Tensor, z_traj: Tensor, group_ids: Tensor, temperature: float = 0.07
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Symmetric InfoNCE over groups of counterfactuals of the same scene.

    Samples sharing a group id are K counterfactual variants of the same camera
    frame that differ only in their instruction (and therefore their target
    trajectory). All K instruction embeddings are scored against all K trajectory
    embeddings with a dot product, giving a K x K score matrix per group.
    Cross-entropy pushes the diagonal (the true pairings) up in both directions:
      rows:    instruction -> which trajectory follows it (forward alignment)
      columns: trajectory  -> which instruction explains it (inverse head)
    Since the image is identical within a group, the instruction is the only
    signal that can solve the matching, forcing reasoning-action binding.

    Args:
        z_text: [B, D] L2-normalised instruction embeddings.
        z_traj: [B, D] L2-normalised trajectory embeddings.
        group_ids: [B] int64; samples sharing an id form one group.
        temperature: softmax temperature of the InfoNCE loss.

    Returns:
        loss: [B] per-sample loss, 0 for singleton groups.
        count: [B] int64, 1 where the sample took part in the loss.
        accuracy: mean text->trajectory retrieval accuracy (None if all groups are singletons).
    """
    loss = z_text.new_zeros(z_text.size(0))
    count = torch.zeros_like(group_ids)
    accuracies = []
    for group_id in group_ids.unique():
        idx = (group_ids == group_id).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue  # no counterfactual siblings to contrast against
        logits = z_text[idx] @ z_traj[idx].T / temperature  # [K, K]
        labels = torch.arange(idx.numel(), device=logits.device)
        loss_fwd = F.cross_entropy(logits, labels, reduction="none")    # instruction -> trajectory
        loss_inv = F.cross_entropy(logits.T, labels, reduction="none")  # trajectory -> instruction
        loss[idx] = 0.5 * (loss_fwd + loss_inv)
        count[idx] = 1
        accuracies.append((logits.argmax(-1) == labels).float().mean())
    accuracy = torch.stack(accuracies).mean() if accuracies else None
    return loss, count, accuracy

def grouped_rank_cycle_loss(
    ce: Tensor, pair_row: Tensor, pair_col: Tensor, batch_size: int, temperature: float = 1.0
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Within-group ranking of trajectories against the instructions that explain
    them, from precomputed per-pair language cross-entropies.

    ce[p] is the mean-token CE of candidate instruction pair_col[p] teacher-forced
    behind the (vision-free) encoding of trajectory pair_row[p]. Arranged as a
    per-group K x K matrix M[i][j], the objective is a symmetric within-group
    softmax over -M/temperature: rows ask "which instruction explains this
    trajectory", columns ask "which trajectory does this instruction explain".
    Ranking (not absolute CE) removes gradient pressure on tokens a trajectory
    cannot legitimately encode (scene references, template phrasing shared by
    all siblings).

    M[i][j] is dominated by an intrinsic per-instruction language-model prior:
    the same text j scored behind any trajectory costs roughly the same, and
    that offset is identical down column j. A raw row softmax is NOT invariant
    to a per-column offset, so it just ranks instructions by how cheap they are
    to say - the same winner for every trajectory in the group, i.e. exactly
    1/K correct with a loss ABOVE ln(K). Columns are centred first, which
    cancels the prior exactly and leaves only the trajectory-conditional part;
    the column direction is already invariant to it (softmax is shift-invariant
    along the axis it normalises), so centring changes only the row term.

    Args:
        ce: [P] per-pair mean-token cross-entropies.
        pair_row: [P] int64 batch index of the trajectory in each pair.
        pair_col: [P] int64 batch index of the candidate instruction.
        batch_size: number of samples in the flattened batch.
        temperature: softmax temperature over -CE.

    Returns:
        loss: [batch_size] per-sample loss, 0 for samples with no siblings.
        count: [batch_size] int64, 1 where the sample took part in the loss.
        accuracy: mean top-1 self-explanation accuracy (None if no pairs).
    """
    loss = ce.new_zeros(batch_size)
    count = torch.zeros(batch_size, dtype=torch.long, device=ce.device)
    correct = []

    position = {
        (int(r), int(c)): p
        for p, (r, c) in enumerate(zip(pair_row.tolist(), pair_col.tolist()))
    }
    # every member of a complete group yields the same sorted member tuple
    groups = {tuple(sorted(pair_col[pair_row == i].tolist())) for i in pair_row.unique().tolist()}

    for members in sorted(groups):
        k = len(members)
        if k < 2:
            continue  # no counterfactual siblings to rank against
        if any((r, c) not in position for r in members for c in members):
            continue  # incomplete K x K block, cannot rank
        flat = ce[torch.tensor([position[(r, c)] for r in members for c in members], device=ce.device)]
        m = flat.view(k, k)

        m = m - m.mean(dim=0, keepdim=True)  # drop the per-instruction prior
        logits = -m / temperature
        labels = torch.arange(k, device=ce.device)
        row_loss = F.cross_entropy(logits, labels, reduction="none")
        col_loss = F.cross_entropy(logits.t(), labels, reduction="none")

        idx = torch.tensor(members, device=ce.device)
        loss[idx] = 0.5 * (row_loss + col_loss)
        count[idx] = 1
        correct.append((logits.argmax(dim=1) == labels).float())

    accuracy = torch.cat(correct).mean() if correct else None
    return loss, count, accuracy


def group_delta_spans(cand_id_list: list) -> list:
    """
    Per-candidate token spans that actually discriminate within a group.

    Sibling instructions in a dreamer group share a template (common prefix,
    often a common suffix); only a short middle clause differs. Scoring the
    mean-token CE over the whole candidate buries that clause under tokens
    every sibling shares. This trims the longest common prefix and suffix
    across all K candidates and returns the remaining [start, end) span per
    candidate, falling back to the full span whenever the trim would leave
    nothing (identical candidates, singletons).

    Args:
        cand_id_list: list of K 1-D int tensors, the group's candidate token
            sequences (variable length).

    Returns:
        list of K (start, end) tuples, end > start guaranteed.
    """
    K = len(cand_id_list)
    lens = [c.size(0) for c in cand_id_list]
    if K < 2:
        return [(0, lens[0])] if K else []
    min_len = min(lens)
    lcp = 0
    while lcp < min_len and all(
        int(c[lcp]) == int(cand_id_list[0][lcp]) for c in cand_id_list[1:]
    ):
        lcp += 1
    # bounded so the suffix can never overlap the prefix on the shortest candidate
    max_lcs = min_len - lcp
    lcs = 0
    while lcs < max_lcs and all(
        int(c[c.size(0) - 1 - lcs]) == int(cand_id_list[0][cand_id_list[0].size(0) - 1 - lcs])
        for c in cand_id_list[1:]
    ):
        lcs += 1
    spans = []
    for c in cand_id_list:
        start, end = lcp, c.size(0) - lcs
        if end <= start:
            start, end = 0, c.size(0)
        spans.append((start, end))
    return spans


def summarise_losses(
    loss_dict: Dict[str, Tuple[Tensor, Tensor]], weights: Optional[Dict[str, float]] = None
) -> TrainingOutput:
    """
    Computes the total loss from a dictionary of losses and their counts.

    The loss dict should contain two tensor for each key:
    - The loss value for each batch sample; shape [B].
    - The loss count for each batch sample; shape [B]. This is the number of items to average over, i.e.
      number of tokens, number of cuboids etc. For the case where each batch sample has a loss, you
      can set it to a ones tensor of shape [B].

    Optionally, a weights dictionary can be provided to weight the losses.

    Args:
        loss_dict: A dictionary of losses and their counts, for each batch sample.
        weights: A dictionary of weights for each loss key.

    Returns:
        A TrainingOutput object with the total loss and its components.
    """

    loss_values = {k: v for k, (v, _) in loss_dict.items()}
    loss_counts = {k: n for k, (_, n) in loss_dict.items()}
    loss_averages = {k: torch.where(n.sum() > 0, v.sum() / n.sum(), 0.0) for k, (v, n) in loss_dict.items()}
    if weights is None:
        loss = torch.stack(list(loss_averages.values())).sum()
    else:
        loss = torch.stack([weights.get(k, 1.0) * v for k, v in loss_averages.items()]).sum()
    return TrainingOutput(
        loss=loss,
        loss_values=loss_values,
        loss_counts=loss_counts,
        loss_averages=loss_averages,
    )