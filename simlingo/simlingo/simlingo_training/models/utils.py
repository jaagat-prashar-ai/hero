
import math

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch import Tensor
from simlingo_training.utils.custom_types import TrainingOutput


def periodic_auxiliary_active(step: int, probability: float) -> bool:
    """Deterministic, DDP-safe whole-batch modality-dropout schedule.

    Every rank sees the same ``global_step`` and therefore takes the same
    forward path.  This avoids rank-local Bernoulli draws changing the DDP
    collective sequence while preserving the requested long-run frequency.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {probability}")
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    period = max(1, round(1.0 / probability))
    return int(step) % period == 0


def diagonal_gaussian_kl(
    mean_p: Tensor,
    mean_q: Tensor,
    *,
    sigma_p: float,
    sigma_q: float,
    detach_q: bool = True,
) -> Tensor:
    """Per-sample KL(N(mean_p,sigma_p²) || N(mean_q,sigma_q²)).

    The trajectory heads emit deterministic coordinates.  Treating those
    coordinates as Gaussian means gives the vision-full and instruction-only
    paths honest distributions without applying a meaningless softmax over
    x/y coordinates.  ``sigma_q`` is intentionally broader: instruction-only
    behavior is ambiguous and should be a prior, not an exact path teacher.
    """
    if sigma_p <= 0.0 or sigma_q <= 0.0:
        raise ValueError("Gaussian sigmas must be positive")
    if mean_p.shape != mean_q.shape:
        raise ValueError(f"KL mean shapes differ: {mean_p.shape} vs {mean_q.shape}")
    q = mean_q.detach() if detach_q else mean_q
    p = mean_p.float().flatten(1)
    q = q.float().flatten(1)
    var_p = float(sigma_p) ** 2
    var_q = float(sigma_q) ** 2
    elementwise = (
        torch.log(p.new_tensor(float(sigma_q) / float(sigma_p)))
        + (var_p + (p - q).square()) / (2.0 * var_q)
        - 0.5
    )
    return elementwise.mean(dim=1)


def mixture_trajectory_nll(
    means: Tensor, log_sigmas: Tensor, mode_logits: Tensor, target: Tensor
) -> Tensor:
    """Per-sample NLL of target under a K-mode diagonal-Gaussian trajectory mixture.

    means/log_sigmas: [B, K, N, D]; mode_logits: [B, K]; target: [B, N, D].
    Returns [B]. The log-sum-exp over modes means only the modes near the GT
    need to explain it -- the standard MDN objective that lets distinct modes
    specialise to distinct valid plans instead of averaging them.
    """
    if means.shape != log_sigmas.shape:
        raise ValueError(f"means/log_sigmas shapes differ: {means.shape} vs {log_sigmas.shape}")
    if means.shape[0] != target.shape[0] or means.shape[2:] != target.shape[1:]:
        raise ValueError(f"target shape {target.shape} incompatible with means {means.shape}")
    m = means.float()
    ls = log_sigmas.float()
    t = target.float().unsqueeze(1)
    component_ll = (
        -0.5 * ((t - m) / ls.exp()).square() - ls - 0.5 * math.log(2.0 * math.pi)
    ).flatten(2).sum(-1)
    log_pi = F.log_softmax(mode_logits.float(), dim=-1)
    return -torch.logsumexp(log_pi + component_ll, dim=1)


def _pairwise_diagonal_gaussian_kl(
    mean_p: Tensor, log_sigma_p: Tensor, mean_q: Tensor, log_sigma_q: Tensor
) -> Tensor:
    """KL between every p component and every q component.

    mean/log_sigma p: [B, Kp, N]; q: [B, Kq, N] (flattened dims). Returns [B, Kp, Kq].
    """
    mp, lsp = mean_p.unsqueeze(2), log_sigma_p.unsqueeze(2)
    mq, lsq = mean_q.unsqueeze(1), log_sigma_q.unsqueeze(1)
    var_p = (2.0 * lsp).exp()
    var_q = (2.0 * lsq).exp()
    return ((lsq - lsp) + (var_p + (mp - mq).square()) / (2.0 * var_q) - 0.5).sum(-1)


def mixture_kl_variational(
    p_means: Tensor,
    p_log_sigmas: Tensor,
    p_logits: Tensor,
    q_means: Tensor,
    q_log_sigmas: Tensor,
    q_logits: Tensor,
    detach_q: bool = True,
) -> Tensor:
    """Hershey-Olsen variational approximation of KL(p || q) between two
    diagonal-Gaussian mixtures over trajectories. Returns [B].

    Deterministic and closed-form (K^2 pairwise Gaussian KLs, no sampling).
    Zero-forcing in the mixture sense: p pays heavily for mass where q is
    unlikely, so a camera plan leaving the text-licensed region is expensive
    while agreeing with ANY text mode is cheap. Sharpening inside a broad q
    mode is not free -- the per-dimension log sigma-ratio is a mild constant
    pressure against over-confidence -- but it is dominated by the mean-
    displacement term. The approximation can dip slightly negative, so it is
    clamped at 0.
    """
    if detach_q:
        q_means, q_log_sigmas, q_logits = (
            q_means.detach(), q_log_sigmas.detach(), q_logits.detach()
        )
    pm, pls = p_means.float().flatten(2), p_log_sigmas.float().flatten(2)
    qm, qls = q_means.float().flatten(2), q_log_sigmas.float().flatten(2)
    log_pi_p = F.log_softmax(p_logits.float(), dim=-1)
    log_pi_q = F.log_softmax(q_logits.float(), dim=-1)
    kl_pp = _pairwise_diagonal_gaussian_kl(pm, pls, pm, pls)
    kl_pq = _pairwise_diagonal_gaussian_kl(pm, pls, qm, qls)
    numer = torch.logsumexp(log_pi_p.unsqueeze(1) - kl_pp, dim=2)
    denom = torch.logsumexp(log_pi_q.unsqueeze(1) - kl_pq, dim=2)
    kl = (log_pi_p.exp() * (numer - denom)).sum(dim=1)
    return kl.clamp_min(0.0)


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
    # zero for singletons but tied to both embeddings, so every rank's backward
    # reaches the same parameters whatever its group composition
    loss = (z_text * 0.0).sum(-1) + (z_traj * 0.0).sum(-1)
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


def apply_traj_controls(traj_pts: Tensor, noise_m: float = 0.0, shuffle: bool = False) -> Tensor:
    """
    Probe-family rigor controls applied to the cycle pass's trajectory input.

    noise_m > 0 adds isotropic Gaussian noise (meters, matching the waypoint
    units) to every point — probes whether the ranking signal survives losing
    sub-noise-scale geometry. shuffle=True re-draws a batch-level permutation
    of the trajectories EVERY call, so no consistent trajectory<->instruction
    pairing exists for the loss to learn: a genuine signal must collapse to
    chance under it, and anything above chance is leakage through a
    trajectory-independent channel (span positions, lengths, group sizes).

    Order matters only in that shuffle is applied last; both default to no-op.
    """
    if noise_m > 0:
        traj_pts = traj_pts + torch.randn_like(traj_pts) * noise_m
    if shuffle:
        traj_pts = traj_pts[torch.randperm(traj_pts.size(0), device=traj_pts.device)]
    return traj_pts


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
    # clamp the denominator: torch.where backprops through BOTH branches, so an
    # unclamped v.sum()/0 emits NaN gradients even when the 0.0 branch is taken.
    # Any loss vector tied into the model graph (e.g. the cycle loss's
    # pair-less-rank wp_encoder tie-in) then floods the shared parameters -
    # and every other rank via the gradient allreduce - with NaN.
    loss_averages = {
        k: torch.where(n.sum() > 0, v.sum() / n.sum().clamp(min=1), v.sum() * 0.0)
        for k, (v, n) in loss_dict.items()
    }
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
