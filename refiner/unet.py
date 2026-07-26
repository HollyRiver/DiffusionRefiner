"""Self-contained RefinementUNet (residual-diffusion refiner R) for baseline_ver03.

Ported from ST-DiT/stdit/models/refinement_unet.py (class RefinementUNet) and its
building blocks from ST-DiT/stdit/models/conv_dit_unet.py
(TimestepEmbedder, SeasonalEmbedder, LeadTimeEmbedder, ResBlock, Downsample, Upsample).

This file has NO `stdit` dependency — plain torch only. It is added under
baseline_ver03/ per the "add new files only" constraint; nothing in baseline_ver03's
core is modified.

The R model refines a blurry base prediction `y_blur` by predicting the clean
residual `x0 = GT_target - y_blur` via x0-parameterized diffusion:

    forward(x, x_context, t, day, hour, lead_time_idx, y_blur) -> predicted x0, (B, 1, H, W)

Input channels stacked internally: 1 (noisy residual) + T context + 1 (y_blur) = 7 (T=5).
Conditioning: t + day/hour + lead_time_idx summed to a hidden vector, injected into
each ResBlock (FiLM-style additive timestep projection).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# Building blocks (copied from conv_dit_unet.py, plain-torch)
# ═════════════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """DDPM-style residual conv block with FiLM-style timestep modulation."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_groups: int = 32):
        super().__init__()
        ng = min(num_groups, in_ch) if in_ch >= num_groups else max(1, in_ch // 4)
        self.norm1 = nn.GroupNorm(ng, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # FiLM: add timestep projection (broadcast over spatial)
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Strided conv downsample (learnable)."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Bilinear upsample + conv (artifact-free)."""

    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep + 2-layer MLP (DiT/DDPM standard)."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embed(t, self.frequency_dim)
        return self.mlp(emb)


class SeasonalEmbedder(nn.Module):
    """day-of-year + hour-of-day cyclic encoding -> MLP."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_size),  # (cos_d, sin_d, cos_h, sin_h)
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, day: torch.Tensor, hour: torch.Tensor) -> torch.Tensor:
        day_rad = day.float() * (2.0 * math.pi / 366.0)
        hour_rad = hour.float() * (2.0 * math.pi / 24.0)
        feat = torch.stack([torch.cos(day_rad), torch.sin(day_rad),
                            torch.cos(hour_rad), torch.sin(hour_rad)], dim=-1)
        return self.mlp(feat)


class LeadTimeEmbedder(nn.Module):
    """Lead time index -> embedding (continuous via small MLP)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, lead_time_idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(lead_time_idx.float().unsqueeze(-1))


# ═════════════════════════════════════════════════════════════════════════════
# RefinementUNet
# ═════════════════════════════════════════════════════════════════════════════


class RefinementUNet(nn.Module):
    """Lead-shared refinement diffusion model (predicts clean residual x0).

    Input channels: 1 (noisy_residual) + T (context) + 1 (y_blur) = 7 (default T=5).
    Conditioning: t + day + hour + lead_time_idx -> hidden vector -> ResBlock FiLM.
    """

    def __init__(
        self,
        input_size: int = 512,
        num_context: int = 5,
        base_ch: int = 32,
        num_levels: int = 3,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        hidden_size: int = 256,
        num_res_blocks: int = 2,
        num_leads_out: int = 1,
        use_prev_lead_input: bool = False,
        extra_input_ch: int = 0,
    ):
        super().__init__()
        assert len(ch_mult) == num_levels
        self.input_size = input_size
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        self.num_leads_out = num_leads_out
        self.use_prev_lead_input = use_prev_lead_input
        self.extra_input_ch = extra_input_ch
        # noisy_residual + context + y_blur (+ optional y_prev_refined) (+ optional extra_input)
        self.in_channels = (
            num_leads_out + num_context + num_leads_out
            + (1 if use_prev_lead_input else 0) + extra_input_ch
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.seasonal_embedder = SeasonalEmbedder(hidden_size)
        self.lead_embedder = LeadTimeEmbedder(hidden_size)

        self.stem = nn.Conv2d(self.in_channels, base_ch, kernel_size=3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        ch = base_ch
        for i in range(num_levels):
            out_ch = base_ch * ch_mult[i]
            blocks = nn.ModuleList([
                ResBlock(ch if k == 0 else out_ch, out_ch, time_emb_dim=hidden_size)
                for k in range(num_res_blocks)
            ])
            self.enc_blocks.append(blocks)
            self.enc_downs.append(Downsample(out_ch))
            ch = out_ch

        # Mid
        self.mid_block = ResBlock(ch, ch, time_emb_dim=hidden_size)

        # Decoder
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(num_levels)):
            skip_ch = base_ch * ch_mult[i]
            self.dec_ups.append(Upsample(ch))
            blocks = nn.ModuleList([
                ResBlock(ch + skip_ch if k == 0 else (base_ch * ch_mult[max(0, i - 1)] if i > 0 else base_ch),
                         base_ch * ch_mult[max(0, i - 1)] if i > 0 else base_ch,
                         time_emb_dim=hidden_size)
                for k in range(num_res_blocks)
            ])
            self.dec_blocks.append(blocks)
            ch = base_ch * ch_mult[max(0, i - 1)] if i > 0 else base_ch

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, num_leads_out, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _make_cond(self, t, day, hour, lead_time_idx):
        return self.t_embedder(t) + self.seasonal_embedder(day, hour) + self.lead_embedder(lead_time_idx)

    def forward(
        self,
        x: torch.Tensor,                   # (B, 1, H, W) noisy residual
        x_context: torch.Tensor,           # (B, T, 1, H, W) or (B, T, H, W)
        t: torch.Tensor,
        day: torch.Tensor,
        hour: torch.Tensor,
        lead_time_idx: torch.Tensor,
        y_blur: torch.Tensor,              # (B, 1, H, W) — base's blurry prediction
        time_diffs: torch.Tensor | None = None,
        x0_prev: torch.Tensor | None = None,
        y_prev_refined: torch.Tensor | None = None,
        extra_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.size(0)
        # context flatten: (B, T, 1, H, W) -> (B, T, H, W)
        if x_context.dim() == 5:
            x_ctx_flat = x_context.squeeze(2) if x_context.size(2) == 1 else x_context.view(B, -1, x.size(2), x.size(3))
        else:
            x_ctx_flat = x_context
        # Stack input channels
        parts = [x, x_ctx_flat, y_blur]
        if self.use_prev_lead_input:
            if y_prev_refined is None:
                y_prev_refined = torch.zeros(B, 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            parts.append(y_prev_refined)
        if self.extra_input_ch > 0:
            if extra_input is None:
                extra_input = torch.zeros(B, self.extra_input_ch, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            parts.append(extra_input)
        h = torch.cat(parts, dim=1)
        assert h.size(1) == self.in_channels, f"expected {self.in_channels} channels, got {h.size(1)}"

        cond = self._make_cond(t, day, hour, lead_time_idx)
        h = self.stem(h)

        skips = []
        for i in range(self.num_levels):
            for block in self.enc_blocks[i]:
                h = block(h, cond)
            skips.append(h)
            h = self.enc_downs[i](h)

        h = self.mid_block(h, cond)

        for j, i in enumerate(reversed(range(self.num_levels))):
            h = self.dec_ups[j](h)
            skip = skips[self.num_levels - 1 - j]
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in self.dec_blocks[j]:
                h = block(h, cond)

        return self.out_conv(F.silu(self.out_norm(h)))


def build_refiner_unet(
    input_size: int = 512,
    num_context: int = 5,
    base_ch: int = 32,
    num_levels: int = 3,
    ch_mult: tuple[int, ...] = (1, 2, 4),
    hidden_size: int = 256,
    num_res_blocks: int = 2,
    num_leads_out: int = 1,
) -> RefinementUNet:
    """Build R with the task defaults."""
    return RefinementUNet(
        input_size=input_size,
        num_context=num_context,
        base_ch=base_ch,
        num_levels=num_levels,
        ch_mult=ch_mult,
        hidden_size=hidden_size,
        num_res_blocks=num_res_blocks,
        num_leads_out=num_leads_out,
    )


class NoiseSchedule:
    """Linear beta schedule for residual x0-diffusion.

    T=1000, beta 1e-4 -> 2e-2 (linear). add_noise(x0,t,noise) = sqrt(abar)*x0 + sqrt(1-abar)*noise.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device=None):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def sample_t(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.T, (batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_a = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_om = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_om * noise


@torch.no_grad()
def r_ddim_refine(
    R: RefinementUNet,
    context: torch.Tensor,
    y_blur: torch.Tensor,
    day: torch.Tensor,
    hour: torch.Tensor,
    lead_idx: torch.Tensor,
    noise_sched: NoiseSchedule,
    num_steps: int = 40,
    init_noise: torch.Tensor | None = None,
    time_diffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """DDIM (x0-param) sampling of the residual. Returns y_sharp = y_blur + residual.

    Ported from eval_refinement.py::r_ddim_refine. eta=0 deterministic DDIM,
    timesteps high->low.
    """
    B, _, H, W = y_blur.shape
    device = y_blur.device
    img = init_noise.clone() if init_noise is not None else torch.randn(B, 1, H, W, device=device)
    step_ratio = noise_sched.T // num_steps
    timesteps = torch.arange(0, num_steps, device=device) * step_ratio
    timesteps = timesteps.flip(0)  # high -> low

    for i, step in enumerate(timesteps):
        t = torch.full((B,), int(step.item()), device=device, dtype=torch.long)
        prev_step = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0, device=device)

        pred_x0 = R(img, context, t, day, hour, lead_idx, y_blur, time_diffs)

        alpha_t = noise_sched.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t_prev = (noise_sched.alphas_cumprod[torch.full((B,), int(prev_step.item()), device=device, dtype=torch.long)].view(-1, 1, 1, 1)
                        if int(prev_step.item()) > 0 else torch.ones_like(alpha_t))
        pred_eps = (img - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt()
        img = alpha_t_prev.sqrt() * pred_x0 + (1 - alpha_t_prev).sqrt() * pred_eps

    return y_blur + img
