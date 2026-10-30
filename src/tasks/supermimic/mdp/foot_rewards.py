# Adapted from RoboParty/P-BFM (2aa985ac50162e446532d1e7231f81902c221e49).
# CC BY-NC 4.0; see ../perception/LICENSE-P-BFM and THIRD_PARTY_NOTICES.md.
"""Tensor-only foot costs. These require measured contacts/terrain, not just FK.

The manager adapters live in rewards.py. Values are nonnegative costs; use
negative weights when a learned counterpart is available for imagination.
"""
import torch

# Canonical output/replay order. Values are costs, without reward weights or dt.
FOOT_REWARD_NAMES = ("feet_stumble", "feet_at_plane")

HEIGHT_SCAN_MIN = -5.0
HEIGHT_SCAN_MAX = 5.0

def clipped_height_clearance(
    sample_height: torch.Tensor,
    hit_height: torch.Tensor,
    distance: torch.Tensor,
    *,
    offset: float = 0.0,
) -> torch.Tensor:
    """Return finite sample-to-terrain clearance, filling invalid rays with the clip maximum."""

    clearance = sample_height - hit_height - float(offset)
    valid = (distance >= 0.0) & torch.isfinite(distance) & torch.isfinite(clearance)
    clearance = torch.where(valid, clearance, HEIGHT_SCAN_MAX)
    return clearance.clamp(HEIGHT_SCAN_MIN, HEIGHT_SCAN_MAX)


def fixed_sole_weights(
    local_xy: torch.Tensor,
    edge_weight: float = 0.1,
    center_weight: float = 1.0,
) -> torch.Tensor:
    """Return a terrain-independent 2-D sole-edge-to-center weight map."""

    if local_xy.ndim != 2 or local_xy.shape[-1] != 2 or local_xy.shape[0] == 0:
        raise ValueError(f"local_xy must be a non-empty (points, 2) tensor, got {tuple(local_xy.shape)}")
    if not 0.0 <= edge_weight <= center_weight:
        raise ValueError(
            f"Expected 0 <= edge_weight <= center_weight, got {edge_weight} and {center_weight}"
        )

    def axis_center_profile(values: torch.Tensor) -> torch.Tensor:
        span = values.max() - values.min()
        if float(span) <= 0.0:
            return torch.ones_like(values)
        fraction = ((values - values.min()) / span).clamp(0.0, 1.0)
        profile = 1.0 - 2.0 * torch.abs(fraction - 0.5)
        return profile / profile.max().clamp_min(torch.finfo(values.dtype).eps)

    x_profile = axis_center_profile(local_xy[:, 0])
    y_profile = axis_center_profile(local_xy[:, 1])
    center_profile = torch.minimum(x_profile, y_profile)
    return edge_weight + (center_weight - edge_weight) * center_profile


def feet_stumble_penalty(
    force_history: torch.Tensor,
    horizontal_ratio: float = 1.0,
    force_threshold: float = 20.0,
    force_scale: float = 100.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize the largest horizontal-dominant foot contact in a policy step."""

    if force_history.ndim != 4 or force_history.shape[-1] != 3:
        raise ValueError(f"Expected force history shaped (N, contacts, history, 3), got {tuple(force_history.shape)}")
    if force_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("force_scale and max_penalty must be positive")
    if force_history.shape[1] == 0 or force_history.shape[2] == 0:
        return torch.zeros(force_history.shape[0], dtype=force_history.dtype, device=force_history.device)
    force_xy = torch.linalg.norm(force_history[..., :2], dim=-1)
    force_z = torch.abs(force_history[..., 2])
    excess = torch.relu(force_xy - horizontal_ratio * force_z - force_threshold)
    return torch.clamp(excess / force_scale, max=max_penalty).amax(dim=(1, 2))


def _support_deficiency(
    clearance: torch.Tensor,
    point_weights: torch.Tensor | None,
    *,
    height_tolerance: float,
    transition_width: float,
) -> torch.Tensor:
    width = max(float(transition_width), 1.0e-6)
    support = torch.sigmoid((height_tolerance - clearance) / width)
    unsupported = 1.0 - support
    boundary_eps = 8.0 * torch.finfo(clearance.dtype).eps
    unsupported = torch.where(
        clearance > height_tolerance + boundary_eps,
        unsupported,
        torch.zeros_like(unsupported),
    )
    if point_weights is None:
        return unsupported.mean(dim=-1)
    weights = point_weights.to(device=unsupported.device, dtype=unsupported.dtype)
    return (unsupported * weights).sum(dim=-1) / weights.sum().clamp_min(torch.finfo(unsupported.dtype).eps)


def feet_at_plane_penalty(
    clearance: torch.Tensor,
    is_contact: torch.Tensor,
    point_weights: torch.Tensor,
    *,
    height_tolerance: float = 0.030,
    transition_width: float = 0.005,
) -> torch.Tensor:
    """Penalize unsupported sole area for both feet on every terrain type."""

    if clearance.ndim != 3 or is_contact.shape != clearance.shape[:2]:
        raise ValueError(
            f"Expected clearance=(N, 2, points), is_contact=(N, 2), got "
            f"{tuple(clearance.shape)}, {tuple(is_contact.shape)}"
        )
    if point_weights.ndim != 1 or point_weights.numel() != clearance.shape[-1]:
        raise ValueError(f"Expected one weight per terrain point, got {tuple(point_weights.shape)}")
    weighted = _support_deficiency(
        clearance,
        point_weights,
        height_tolerance=height_tolerance,
        transition_width=transition_width,
    )
    return torch.sum(weighted * is_contact.to(weighted.dtype), dim=1)
