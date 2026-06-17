#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoundCause inference-only model code.

Loads the pretrained checkpoint and exposes:
    - ModelConfig:  configuration class
    - CausalDiscoveryTransformer: the model (139M parameters)
    - predict(model, X, device, ...): run inference on observational data

Released for inference only.  The training loop, synthetic data generator,
and internal infrastructure integrations are not included.

Usage:
    import torch
    from foundcause import ModelConfig, CausalDiscoveryTransformer, predict

    cfg = ModelConfig()
    model = CausalDiscoveryTransformer(cfg)
    ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # X: numpy array (N, D), N samples, D variables (2 <= D <= 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    result = predict(model, X, device)
    adjacency = result["adjacency"]  # (D, D) binary DAG
    probabilities = result["probabilities"]  # (D, D) edge probs
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class ModelConfig:
    """Model architecture configuration (inference-only subset)."""
    hidden_dim: int = 768
    num_encoder_layers: int = 8      # Total attention blocks = 2 * this
    num_heads: int = 8
    num_confounder_tokens: int = 8
    logit_bias_init: float = -2.0    # sigmoid(-2) ~= 0.12 -- sparsity prior

    num_tri_refine_rounds: int = 3   # triangular edge refinement rounds (0 disables)
    tri_pair_dim: int = 192          # hidden dim for pair features in triangular refinement
    pma_num_queries: int = 8         # PMA pooling: number of learned query tokens
    pma_num_heads: int = 4           # PMA pooling: number of attention heads
    predict_confounders: bool = True # False disables the confounder head
    feat_drop_asym: float = 0.2      # no-op at inference; kept for API compatibility
    disable_pairwise_stats: bool = False  # True disables all 45 hand-crafted pairwise features

    # Kept for compatibility with the forward pass; acyclicity is not enforced
    # at inference (lambda_acyc is effectively 0).
    lambda_acyc: float = 0.0
    acyclicity_power_iters: int = 10


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(x.dtype) * self.scale


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer, 2020). Parameter-neutral vs ReLU at 2/3 width."""
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = int(dim * mult * 2 / 3)
        self.w1 = nn.Linear(dim, inner)
        self.w_gate = nn.Linear(dim, inner)
        self.w2 = nn.Linear(inner, dim)

    def forward(self, x):
        return self.w2(torch.nan_to_num((F.silu(self.w_gate(x)) * self.w1(x)).clamp(-10000, 10000), nan=0.0, posinf=10000.0, neginf=-10000.0))


class AttentionBlock(nn.Module):
    """Pre-norm attention + SwiGLU FFN with RMSNorm and FlashAttention."""
    def __init__(self, dim, num_heads, ffn_mult=4, num_blocks=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln_attn = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ln_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_mult)
        # Scaled residual init: softer scaling to avoid dead logits early
        resid_scale = num_blocks ** -0.5
        nn.init.normal_(self.out_proj.weight, std=0.02 * resid_scale)
        nn.init.normal_(self.ffn.w2.weight, std=0.02 * resid_scale)

    def forward(self, z, key_padding_mask=None, stat_bias=None):
        B, S, D = z.shape
        h, d = self.num_heads, self.head_dim
        normed = self.ln_attn(z)
        q = self.q_proj(normed).reshape(B, S, h, d).transpose(1, 2)
        kv = self.kv_proj(normed).reshape(B, S, 2, h, d).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S) bool
        if stat_bias is not None:
            # stat_bias: (B_orig, heads, D, D) -> expand to (B*N, heads, D, D)
            B_orig = stat_bias.shape[0]
            N_expand = B // B_orig  # B here is B*N from reshape
            sb = stat_bias.unsqueeze(1).expand(-1, N_expand, -1, -1, -1).reshape(B, self.num_heads, S, S)
            if attn_mask is not None:
                # Combine bool padding mask with stat bias (single allocation)
                attn_mask = sb.masked_fill(~attn_mask.expand_as(sb), -1e9)
            else:
                attn_mask = sb
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        z = z + self.out_proj(out.transpose(1, 2).reshape(B, S, D))
        z = z + self.ffn(self.ln_ffn(z))
        # Guard against NaN from SDPA/inductor at large D under compile+bf16
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z


class PMAPooling(nn.Module):
    """Pooling by Multihead Attention (Lee et al., Set Transformer, ICML 2019).

    K learned query tokens cross-attend into per-sample representations,
    producing K summary vectors per variable. These are concatenated and
    projected to the final embedding dimension. A gated residual to
    max-pool lets the model fall back to a simple summary when needed.
    """
    def __init__(self, hidden_dim, num_queries=8, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Learned query tokens (shared across all variables)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        # Manual Q/K/V projections for F.scaled_dot_product_attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj_attn = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Project K*H -> H
        self.out_proj = nn.Linear(num_queries * hidden_dim, hidden_dim)
        # Residual gate: blend PMA output with max-pool fallback
        # Initialized near 0 so model starts ≈ max-pool
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    def forward(self, z, mask, var_mask):
        """z: (B, N, D, H), mask: (B, N, D), var_mask: (B, D). Returns (B, D, H)."""
        B, N, D, H = z.shape
        K = self.num_queries

        # Max-pool fallback (always computed for residual)
        # Use -1e4 instead of finfo.min to avoid relying on downstream nan_to_num guard
        mask_exp = mask.unsqueeze(-1).bool()  # (B, N, D, 1)
        z_max = z.masked_fill(~mask_exp, -1e4).max(dim=1).values  # (B, D, H)

        # Reshape for per-variable cross-attention: treat each variable independently
        # z: (B, N, D, H) -> (B*D, N, H)
        z_flat = z.permute(0, 2, 1, 3).contiguous().reshape(B * D, N, H)
        # Key padding mask: (B, N, D) -> (B*D, N)
        kpm = (~mask.bool()).permute(0, 2, 1).contiguous().reshape(B * D, N)
        # Fix: for padded variables (var_mask=0), all samples are masked -> NaN.
        # Unmask all samples for padded vars (output gets zeroed anyway).
        pad_vars = (~var_mask.bool()).unsqueeze(-1).expand(B, D, N).reshape(B * D, N)
        kpm = kpm & ~pad_vars  # unmask padded variables

        # Expand queries: (1, K, H) -> (B*D, K, H)
        q = self.query_norm(self.queries).expand(B * D, -1, -1)

        # Cross-attention via F.scaled_dot_product_attention (FlashAttention compatible)
        h_dim, n_h = self.head_dim, self.num_heads
        q_proj = self.q_proj(q).reshape(B * D, K, n_h, h_dim).transpose(1, 2)  # (B*D, heads, K, h)
        kv = self.kv_proj(z_flat).reshape(B * D, N, 2, n_h, h_dim).permute(2, 0, 3, 1, 4)
        k_proj, v_proj = kv.unbind(0)  # each (B*D, heads, N, h)
        # Boolean mask: True = attend (invert kpm where True = ignore)
        attn_mask = (~kpm).unsqueeze(1).unsqueeze(1)  # (B*D, 1, 1, N) bool
        attn_out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=attn_mask)
        pma_out = self.out_proj_attn(attn_out.transpose(1, 2).reshape(B * D, K, H))  # (B*D, K, H)
        pma_out = pma_out + self.ffn(pma_out)  # (B*D, K, H)

        # Flatten K queries and project: (B*D, K*H) -> (B*D, H)
        pma_flat = pma_out.reshape(B * D, K * H)
        pma_proj = self.out_proj(pma_flat).reshape(B, D, H)  # (B, D, H)

        # Gated residual: blend PMA with max-pool
        g = torch.sigmoid(self.gate)
        out = g * pma_proj + (1 - g) * z_max
        out = torch.nan_to_num(out.clamp(-5000, 5000), nan=0.0, posinf=5000.0, neginf=-5000.0)  # prevent accumulation overflow

        # Zero padded variables
        out = out * var_mask.unsqueeze(-1)
        # Diversity loss (computed here to avoid state mutation that breaks torch.compile)
        q_norm = F.normalize(pma_out.reshape(B, D, K, H), dim=-1)
        cos_sim = torch.bmm(
            q_norm.reshape(B * D, K, H),
            q_norm.reshape(B * D, K, H).transpose(1, 2)
        )
        eye_k = torch.eye(K, device=out.device)
        off_diag = cos_sim * (1 - eye_k).unsqueeze(0)
        div_loss = F.relu(off_diag - 0.5).pow(2).mean()
        return out, div_loss


class StatAttentionBias(nn.Module):
    """Per-layer nonlinear projection of pairwise statistics to attention biases."""
    def __init__(self, num_stats, num_heads, num_layers):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(num_stats, 48), nn.GELU(), nn.Linear(48, num_heads))
            for _ in range(num_layers)
        ])

    def forward(self, stats, layer_idx):
        # Cast to input dtype (bf16) to prevent float32 upcast in attention — values
        # are clamped to [-10, 10] which is safe for bf16 (max ~65504)
        return self.projs[layer_idx](stats).permute(0, 3, 1, 2).clamp(-10, 10).to(stats.dtype)


class Encoder(nn.Module):
    """Alternating axis-swap transformer encoder.

    Even blocks attend across variables (with stat-conditioned attention bias
    and 12 learned global context tokens); odd blocks attend across samples.
    Intermediate RMSNorm every 4 blocks stabilises the residual stream at
    large variable counts. Multi-layer features are fused via learnable
    softmax weights before PMA pooling over the sample axis.
    """
    def __init__(self, hidden_dim, num_layers, num_heads,
                 pma_num_queries=8, pma_num_heads=4, feat_drop_asym=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feat_drop_asym = feat_drop_asym
        self.input_proj = nn.Linear(2, hidden_dim)
        num_blocks = 2 * num_layers
        self.blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, num_blocks=num_blocks)
            for _ in range(num_blocks)])
        # Intermediate RMSNorm every 4 blocks; skip the last so final_norm does not double-normalise.
        _norm_positions = [i for i in range(num_blocks) if (i + 1) % 4 == 0 and i < num_blocks - 1]
        self.intermediate_norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in _norm_positions])
        self._norm_map = {i: idx for idx, i in enumerate(_norm_positions)}
        self.final_norm = RMSNorm(hidden_dim)
        self.pma = PMAPooling(hidden_dim, pma_num_queries, pma_num_heads)
        self.stat_bias = StatAttentionBias(PairwiseStatistics.NUM_FEATURES, num_heads, num_layers)
        # Multi-layer feature fusion: tap every 4th block before the last.
        self._tap_blocks = set(i for i in range(num_blocks) if (i + 1) % 4 == 0 and i < num_blocks - 1)
        self.layer_weights = nn.Parameter(torch.zeros(len(self._tap_blocks) + 1))
        # Global context tokens persist across samples in variable-attention.
        self.num_global_tokens = 12
        self.global_tokens = nn.Parameter(torch.randn(1, 1, self.num_global_tokens, hidden_dim) * 0.02)

    def forward(self, x_val, mask, var_mask, pair_stats=None):
        """
        x_val: (B, N, D) — values, zeros where missing
        mask:  (B, N, D) — 1=observed, 0=missing
        var_mask: (B, D) — 1=real variable, 0=padding
        pair_stats: (B, D, D, NUM_FEATURES) — optional pairwise statistics
        Returns: (B, D, H)
        """
        B, N, D = x_val.shape
        H = self.hidden_dim

        # Per-variable normalization (no user preprocessing needed)
        # Compute in float32 to avoid bfloat16 overflow with extreme values
        with torch.no_grad(), torch.amp.autocast(device_type=x_val.device.type, enabled=False):
            x_f = x_val.float()
            m_f = mask.float()
            obs_count = m_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, D)
            center = (x_f * m_f).sum(dim=1, keepdim=True) / obs_count  # (B, 1, D)
            diff2 = ((x_f - center) * m_f) ** 2
            var = diff2.sum(dim=1, keepdim=True) / (obs_count - 1).clamp(min=1)
            scale = var.sqrt().clamp(min=1e-8)  # (B, 1, D)
            x_normed = ((x_f - center) / scale * m_f).to(x_val.dtype)

        # 2-channel input: shape [..., N, D, 2]
        inp = torch.stack([x_normed, mask], dim=-1)  # (B, N, D, 2)
        z = self.input_proj(inp)  # (B, N, D, H)
        # Zero out embeddings for missing entries so they don't leak
        # a bias-term signal through variable-attention
        z = z.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)  # (B, N, D, H)

        # Key-padding masks
        var_kpm = (~var_mask.bool()).unsqueeze(1).expand(B, N, D).reshape(B * N, D)
        # For sample-attention: padded variables (var_mask=0) have mask=0 for all samples,
        # which would make all keys masked -> NaN from softmax. Fix: unmask all samples
        # for padded variables (their output gets zeroed after max-pool anyway).
        samp_mask = mask.clone()  # (B, N, D)
        pad_vars = (~var_mask.bool()).unsqueeze(1).expand_as(samp_mask)  # (B, N, D)
        samp_mask[pad_vars] = 1.0
        samp_kpm = (~samp_mask.bool()).permute(0, 2, 1).reshape(B * D, N)

        # Alternating blocks with axis swap (swapaxes(-3, -2))
        mask_bool = mask.unsqueeze(-1).bool()  # (B, N, D, 1) for re-zeroing missing entries
        # Stat-conditioned attention bias for variable-attention blocks
        # Stat-conditioned attention biases for variable-attention blocks.
        stat_attn_bias_list = None
        if pair_stats is not None:
            stat_attn_bias_list = [self.stat_bias(pair_stats, l) for l in range(len(self.stat_bias.projs))]
        # Expand global context tokens: (1, 1, K_g, H) -> (B, N, K_g, H)
        K_g = self.num_global_tokens
        g_tok = self.global_tokens.expand(B, N, -1, -1)
        # Extended var_kpm with global tokens always unmasked
        var_kpm_ext = torch.cat([var_kpm, torch.zeros(B * N, K_g, dtype=torch.bool, device=var_kpm.device)], dim=1)
        intermediate_z = []  # collect intermediate representations for multi-layer fusion
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # Concatenate global tokens to variable axis: (B, N, D+K_g, H)
                z_ext = torch.cat([z, g_tok], dim=2)
                z_ext = z_ext.reshape(B * N, D + K_g, H)
                # Extend stat bias if present: pad from (B, heads, D, D) to (B, heads, D+K_g, D+K_g)
                sb = stat_attn_bias_list[i // 2] if stat_attn_bias_list is not None else None
                if sb is not None:
                    sb = F.pad(sb, (0, K_g, 0, K_g), value=0.0)
                z_ext = block(z_ext, key_padding_mask=var_kpm_ext, stat_bias=sb)
                z_ext = z_ext.reshape(B, N, D + K_g, H)
                # Split back: variable embeddings and global tokens
                z = z_ext[:, :, :D, :]
                g_tok = z_ext[:, :, D:, :]
                z = z.masked_fill(~mask_bool, 0.0)  # re-zero missing entries
            else:
                # Attend across samples (sequence = N)
                z = z.permute(0, 2, 1, 3).contiguous()  # (B, D, N, H)
                z = z.reshape(B * D, N, H)
                z = block(z, key_padding_mask=samp_kpm)
                z = z.reshape(B, D, N, H)
                z = z.permute(0, 2, 1, 3).contiguous()  # (B, N, D, H)
                z = z.masked_fill(~mask_bool, 0.0)  # re-zero missing entries
            # Normalize residual stream every 4 blocks to prevent bfloat16 overflow
            if i in self._norm_map:
                z = self.intermediate_norms[self._norm_map[i]](z)
            # Collect intermediate representations for multi-layer fusion
            if i in self._tap_blocks:
                intermediate_z.append(z)

        z = self.final_norm(z)
        # Clamp to prevent bfloat16 overflow from residual accumulation
        # across 16 attention blocks at high variable counts (D=30+)
        z = torch.nan_to_num(z.clamp(-5000, 5000), nan=0.0, posinf=5000.0, neginf=-5000.0)

        # Multi-layer feature fusion: blend intermediate + final representations
        # with learnable softmax weights (initialized uniform)
        if intermediate_z:
            intermediate_z.append(z)  # add final as the last element
            w = torch.softmax(self.layer_weights[:len(intermediate_z)], dim=0)
            z = sum(w[k] * intermediate_z[k] for k in range(len(intermediate_z)))

        # PMA pooling over samples
        z, pma_div = self.pma(z, mask, var_mask)  # (B, D, H)
        return z, pma_div


class PairwiseStatistics(nn.Module):
    """Hand-crafted pairwise features computed under ``no_grad``.

    Produces a ``(B, D, D, 45)`` tensor combining symmetric, antisymmetric,
    V-structure, and reliability-metadata features. See ``forward()`` for
    the per-index layout.
    """

    NUM_FEATURES = 45
    N_ENTROPY_BINS = 20
    M_RFF = 32

    def __init__(self):
        super().__init__()
        self.register_buffer('omega_rff', torch.randn(self.M_RFF))
        self.register_buffer('bias_rff', torch.rand(self.M_RFF) * (2 * 3.14159))

    @torch.compiler.disable
    def forward(self, x_val, mask, var_mask):
        """Returns ``(B, D, D, 45)`` tensor of pairwise statistics.

        Feature indices:
          Symmetric (0-6): corr, skew_prod, partial_corr, spearman,
            nonlin_dep, kurt_prod, quad_dep.
          Antisymmetric (7-36): nw_resid_var (ij, ji), skew/kurt diffs,
            cross-moments, HSIC (ij, ji), residual skew/kurt,
            cond_entropy_asym, igci_asym, precision regression coefficients
            (ij, ji), nonlinearity ratio (ij, ji), local R^2 variance (ij, ji),
            cond_var_cv (ij, ji), prec_diag_asym, prec_rowsum_asym,
            Chatterjee xi (ij, ji), VCM (ij, ji), IGCI variance (ij, ji),
            Stein diagonal variance (differenced).
          V-structure (37-40): collider_evidence, pcorr_gap,
            collider_child (ij, ji).
          Reliability metadata (41-44): log_cond, d_over_n, unique_ratio,
            norm_obs.
        """
        with torch.no_grad():
            B, N, D = x_val.shape

            # Normalize data in float32 to avoid bfloat16 overflow
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
                x_f = x_val.float()
                m_f = mask.float()
                # Robust pre-normalization: use median absolute deviation to handle extreme values
                # that would overflow float32 in variance computation
                obs_count = m_f.sum(dim=1, keepdim=True).clamp(min=1)
                mu = (x_f * m_f).sum(dim=1, keepdim=True) / obs_count
                abs_dev = (x_f - mu).abs() * m_f
                # Use mean absolute deviation * 1.4826 as robust std estimate
                mad = abs_dev.sum(dim=1, keepdim=True) / obs_count * 1.4826
                mad = mad.clamp(min=1e-8)
                # Winsorize to ±10 MAD-sigma before computing actual statistics
                x_f = torch.clamp(x_f, mu - 10 * mad, mu + 10 * mad) * m_f
                # Safety: after winsorization, ensure values are within float32-safe range
                # for squaring (x^2 must not overflow ~3.4e38, so |x| < ~1.8e19)
                # This preserves signal: MAD-winsorization already removed outliers,
                # this just catches edge cases where MAD itself is extreme
                x_f = x_f.clamp(-1e15, 1e15) * m_f
                # Now compute clean stats
                center = (x_f * m_f).sum(dim=1, keepdim=True) / obs_count
                diff2 = ((x_f - center) * m_f) ** 2
                var = diff2.sum(dim=1, keepdim=True) / (obs_count - 1).clamp(min=1)
                scale = var.sqrt().clamp(min=1e-8)
                x = ((x_f - center) / scale * m_f).to(x_val.dtype)
            obs = mask  # (B, N, D)

            # Joint observation counts
            joint_count = torch.bmm(obs.transpose(1, 2), obs)  # (B, D, D)
            joint_count_safe = joint_count.clamp(min=2)

            # ===== SYMMETRIC FEATURES (0-6) =====

            # 0. Pearson correlation
            xx = torch.bmm(x.transpose(1, 2), x)  # (B, D, D)
            corr = xx / (joint_count_safe - 1)  # (B, D, D)
            corr = corr.clamp(-1, 1)

            # 1. Skewness product
            x2 = x ** 2  # (B, N, D) -- reused for cross-moments and kurtosis
            x3 = x * x2  # (B, N, D) -- cheaper than x ** 3
            obs_count_per_var = obs.sum(dim=1).clamp(min=1)  # (B, D)
            skew = (x3 * obs).sum(dim=1) / obs_count_per_var  # (B, D)
            skew = skew.clamp(-5, 5)
            skew_prod = (skew.unsqueeze(2) * skew.unsqueeze(1)).clamp(-5, 5)  # symmetric interaction

            # 2. Partial correlation (PC algorithm signal)
            # Inverse covariance via OAS adaptive shrinkage (Chen et al., 2010)
            cov = corr  # same as xx/(joint_count_safe-1) for standardized data
            eye = torch.eye(D, device=x.device).unsqueeze(0)
            # OAS adaptive shrinkage: computes optimal shrinkage per-graph from trace statistics
            # Mask out padded entries so n_eff / traces reflect only valid variables
            vm_pair = var_mask.unsqueeze(1) * var_mask.unsqueeze(2)  # (B, D, D)
            n_valid_pairs = vm_pair.sum(dim=(1, 2)).clamp(min=1)
            n_eff = (joint_count * vm_pair).sum(dim=(1, 2)) / n_valid_pairs
            n_eff = n_eff.clamp(min=3)  # (B,) effective sample size
            D_cur = var_mask.sum(dim=1).clamp(min=2)  # (B,) effective dimension
            trace_S = (cov.diagonal(dim1=-2, dim2=-1) * var_mask).sum(-1)  # (B,)
            trace_S2 = ((cov ** 2) * vm_pair).sum((-2, -1))  # (B,)
            rho_num = (1 - 2.0 / D_cur) * trace_S2 + trace_S ** 2
            rho_den = (n_eff + 1 - 2.0 / D_cur) * (trace_S2 - trace_S ** 2 / D_cur)
            rho = (rho_num / rho_den.clamp(min=1e-8)).clamp(0.01, 1.0)  # (B,) per-graph, floor at 0.01
            # Shrink toward scaled identity (preserves overall variance level)
            shrink_target = (trace_S / D_cur).view(-1, 1, 1) * eye
            cov_reg = (1 - rho.view(-1, 1, 1)) * cov + rho.view(-1, 1, 1) * shrink_target
            # Zero out padded variables in cov to prevent singular matrix issues
            vm2 = (var_mask.unsqueeze(1) * var_mask.unsqueeze(2))  # (B, D, D)
            cov_reg = cov_reg * vm2 + eye * (1 - vm2)
            # Safety: ensure diagonal is positive (constant-variance columns can produce zero diag)
            cov_reg = cov_reg + 1e-6 * eye
            # Float32 upcast for matrix inverse: bf16 LU decomposition can NaN for condition>100
            prec = torch.linalg.inv(cov_reg.float()).to(cov_reg.dtype)
            prec = torch.nan_to_num(prec, nan=0.0, posinf=0.0, neginf=0.0)
            # Partial corr: -prec[i,j] / sqrt(prec[i,i] * prec[j,j])
            diag_prec = prec.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-8)  # (B, D)
            norm = (diag_prec.unsqueeze(2) * diag_prec.unsqueeze(1)).sqrt()  # (B, D, D)
            partial_corr = (-prec / norm).clamp(-1, 1)  # (B, D, D)

            # 3. Spearman rank correlation (monotonic nonlinear dependence)
            # Jitter-based tie-breaking: add tiny noise to break ties before ranking.
            # Vectorized O(N*D*log(N)) — no Python loops. Jitter magnitude 1e-7 is
            # negligible vs standardized data (std=1) but breaks exact ties from
            # discrete_uniform/discretized variables. Ranks computed in float32 to
            # avoid bfloat16 precision loss for rank values > 128.
            x_for_rank = x.float()  # float32 for rank precision
            x_for_rank = x_for_rank + torch.randn_like(x_for_rank) * 1e-7 * obs.float()
            x_for_rank[~obs.bool()] = float('inf')  # push missing to end of sort
            sorted_idx = x_for_rank.argsort(dim=1)  # (B, N, D)
            ranks = torch.zeros(B, N, D, dtype=torch.float32, device=x.device)
            ranks.scatter_(1, sorted_idx, torch.arange(N, device=x.device).float().view(1, N, 1).expand(B, N, D))
            ranks = ranks * obs.float()  # zero out ranks for missing entries
            # Normalize ranks to zero-mean unit-var per variable using observed entries
            r_count = obs.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, D)
            r_mean = (ranks * obs).sum(dim=1, keepdim=True) / r_count
            r_centered = (ranks - r_mean) * obs
            r_std = (r_centered.pow(2).sum(dim=1, keepdim=True) / (r_count - 1).clamp(min=1)).sqrt().clamp(min=1e-8)
            ranks = r_centered / r_std * obs
            rr = torch.bmm(ranks.transpose(1, 2), ranks)  # (B, D, D)
            spearman = (rr / (joint_count_safe - 1)).clamp(-1, 1)  # (B, D, D)

            # 4. Nonlinear dependence indicator
            nonlin_dep = spearman.abs() - corr.abs()  # [-1, 1]: positive = nonlinear, negative = more linear than monotonic

            # 5. Kurtosis product (joint tail heaviness -- symmetric)
            x4 = x2 ** 2  # (B, N, D)
            kurt = (x4 * obs).sum(dim=1) / obs_count_per_var - 3.0  # excess kurtosis (B, D)
            kurt = kurt.clamp(-5, 100)
            kurt_prod = (kurt.unsqueeze(2) * kurt.unsqueeze(1)).clamp(-500, 10000)  # (B, D, D) symmetric

            # 6. Quadratic cross-dependence
            x2_obs = x2 * obs  # (B, N, D)
            x2x = torch.bmm(x2_obs.transpose(1, 2), x * obs)  # (B, D, D)
            xx2 = x2x.transpose(1, 2)  # xx2[i,j] = x2x[j,i] by symmetry
            quad_corr_ij = (x2x / joint_count_safe).clamp(-5, 5)
            quad_corr_ji = (xx2 / joint_count_safe).clamp(-5, 5)
            quad_dep = ((quad_corr_ij.abs() + quad_corr_ji.abs()) / 2).clamp(0, 5)  # (B, D, D) symmetric

            # ===== ANTISYMMETRIC FEATURES (7-36) =====

            # 9. Skewness difference (skew_i - skew_j)
            skew_diff = (skew.unsqueeze(2) - skew.unsqueeze(1)).clamp(-5, 5)

            # 10. Kurtosis difference (kurt_i - kurt_j)
            kurt_diff = kurt.unsqueeze(2) - kurt.unsqueeze(1)  # (B, D, D)

            # 11-12. Higher-order cross-moments (LiNGAM signal)
            xi2_xj_feat = (x2x / joint_count_safe).clamp(-10, 10)  # E[x_i^2 * x_j]
            xi_xj2_feat = (xx2 / joint_count_safe).clamp(-10, 10)  # E[x_i * x_j^2]

            # 7-8, 13-17. NW kernel regression + HSIC on nonlinear residuals
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
              x_f32 = x.float()
              obs_f32 = obs.float()
              M_rff = self.M_RFF
              omega_rff = self.omega_rff.float()
              bias_rff = self.bias_rff.float()
              sc_rff = (2.0 / M_rff) ** 0.5
              joint_count_f32 = joint_count.float().clamp(min=2)

              # Per-variable median bandwidth
              sigma2_nw = torch.ones(B, D, device=x.device)
              eye_n = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(-1)  # shared across loops
              bw_chunk = min(D, 32)
              for d0 in range(0, D, bw_chunk):
                  d1 = min(d0 + bw_chunk, D)
                  xc = x_f32[:, :, d0:d1]
                  oc = obs_f32[:, :, d0:d1]
                  diff_d = xc.unsqueeze(2) - xc.unsqueeze(1)
                  dist2_d = diff_d ** 2
                  pm = oc.unsqueeze(2) * oc.unsqueeze(1)
                  dist2_d = dist2_d + (1 - pm) * 1e10 + eye_n * 1e10
                  sigma2_nw[:, d0:d1] = dist2_d.permute(0, 3, 1, 2).reshape(
                      B, d1 - d0, N * N).median(dim=-1).values.clamp(min=0.01, max=100)

              # NW regression residuals + HSIC on residuals
              nw_resid_var_ij = torch.zeros(B, D, D, device=x.device)
              nw_resid_skew_ij = torch.zeros(B, D, D, device=x.device)
              nw_resid_kurt_ij = torch.zeros(B, D, D, device=x.device)
              hsic_nw_ij = torch.zeros(B, D, D, device=x.device)
              vcm_ij = torch.zeros(B, D, D, device=x.device)  # Variance of Conditional Mean
              stein_diag_var = torch.zeros(B, D, device=x.device)  # Stein score diagonal variance (per-variable)
              target_var = ((x_f32 ** 2) * obs_f32).sum(dim=1) / (obs_f32.sum(dim=1) - 1).clamp(min=1)
              target_var = target_var.clamp(min=1e-8)  # (B, D)
              x_obs_f32 = x_f32 * obs_f32  # pre-compute once for NW loop
              inv_bw_tgt = 1.0 / (sigma2_nw.sqrt().clamp(min=1e-4) * 0.75)  # hoist: constant across chunks
              eye_n_nw = 1 - eye_n  # re-use eye_n from bandwidth loop

              chunk = min(D, 15)
              for i_start in range(0, D, chunk):
                  i_end = min(i_start + chunk, D)
                  n_c = i_end - i_start
                  x_i = x_f32[:, :, i_start:i_end]
                  obs_i = obs_f32[:, :, i_start:i_end]

                  diff_i = x_i.unsqueeze(2) - x_i.unsqueeze(1)
                  bw = sigma2_nw[:, i_start:i_end]
                  K = torch.exp(-diff_i ** 2 / (2 * bw.unsqueeze(1).unsqueeze(1) + 1e-8))
                  K = K * (obs_i.unsqueeze(2) * obs_i.unsqueeze(1))
                  K = K * eye_n_nw

                  K_flat = K.permute(0, 3, 1, 2).reshape(B * n_c, N, N)
                  x_obs = x_obs_f32.unsqueeze(1).expand(-1, n_c, -1, -1).reshape(B * n_c, N, D)
                  obs_rep = obs_f32.unsqueeze(1).expand(-1, n_c, -1, -1).reshape(B * n_c, N, D)
                  K_sum_flat = torch.bmm(K_flat, obs_rep).clamp(min=1e-8)
                  f_hat = (torch.bmm(K_flat, x_obs) / K_sum_flat).reshape(B, n_c, N, D)

                  # Stein score diagonal: Var[∂²log p/∂x_i²] — density curvature (SCORE algorithm)
                  # Piggyback on NW kernel: J[i,i](x_m) = Σ_n K[m,n]*((x_n-x_m)²/σ⁴ - 1/σ²) / K_sum
                  bw_unsq = bw.unsqueeze(1).unsqueeze(1).clamp(min=1e-4)  # (B, 1, 1, n_c)
                  stein_weights = diff_i ** 2 / (bw_unsq ** 2) - 1.0 / bw_unsq
                  K_sum_per = K.sum(dim=2).clamp(min=1e-8)  # (B, N, n_c)
                  stein_diag = (K * stein_weights).sum(dim=2) / K_sum_per  # (B, N, n_c)
                  stein_diag = stein_diag.clamp(-1000, 1000)
                  stein_diag_var[:, i_start:i_end] = stein_diag.var(dim=1).clamp(0, 50)

                  resid = x_f32.unsqueeze(1) - f_hat
                  jmask = (obs_i.unsqueeze(-1) * obs_f32.unsqueeze(2)).permute(0, 2, 1, 3)
                  resid = resid * jmask
                  jc = joint_count_f32[:, i_start:i_end, :].clamp(min=2)

                  rv = (resid ** 2).sum(dim=2) / (jc - 1).clamp(min=1)
                  nw_resid_var_ij[:, i_start:i_end, :] = (rv / target_var.unsqueeze(1)).clamp(0, 5)

                  rstd = rv.sqrt().clamp(min=1e-8)
                  rn = (resid / rstd.unsqueeze(2)) * jmask
                  rn2 = rn * rn
                  nw_resid_skew_ij[:, i_start:i_end, :] = ((rn * rn2).sum(dim=2) / jc).clamp(-5, 5)
                  nw_resid_kurt_ij[:, i_start:i_end, :] = ((rn2 * rn2).sum(dim=2) / jc - 3).clamp(-5, 100)

                  # VCM: Variance of Conditional Mean — signal strength decomposition
                  # f_hat (B, n_c, N, D) already computed; jmask (B, n_c, N, D) and jc (B, n_c, D) available
                  f_hat_m = f_hat * jmask
                  f_mean = f_hat_m.sum(dim=2) / jc  # (B, n_c, D)
                  vcm_raw = (f_hat_m ** 2).sum(dim=2) / jc - f_mean ** 2  # Var = E[X²] - E[X]²
                  vcm_ij[:, i_start:i_end, :] = (vcm_raw.clamp(min=0) / target_var.unsqueeze(1)).clamp(0, 5)

                  inv_bw = 1.0 / sigma2_nw[:, i_start:i_end].sqrt().clamp(min=1e-4)
                  phi_i = sc_rff * torch.cos(
                      (x_i * inv_bw.unsqueeze(1)).unsqueeze(-1) * omega_rff + bias_rff)
                  phi_i = phi_i * obs_i.unsqueeze(-1)
                  mu_phi_i = phi_i.sum(dim=1) / obs_count_per_var[:, i_start:i_end].unsqueeze(-1)
                  resid_perm = resid.permute(0, 2, 1, 3)
                  jmask_perm = jmask.permute(0, 2, 1, 3)

                  for j0 in range(0, D, chunk):
                      j1 = min(j0 + chunk, D)
                      rc = resid_perm[:, :, :, j0:j1]
                      ibj = inv_bw_tgt[:, j0:j1]
                      phi_r = sc_rff * torch.cos(
                          (rc * ibj.unsqueeze(1).unsqueeze(1)).unsqueeze(-1) * omega_rff + bias_rff)
                      phi_r = phi_r * jmask_perm[:, :, :, j0:j1].unsqueeze(-1)
                      jcj = jc[:, :, j0:j1].unsqueeze(-1).clamp(min=2)
                      cross = (phi_i.unsqueeze(3) * phi_r).sum(dim=1) / jcj
                      mu_r = phi_r.sum(dim=1) / jcj
                      hdiff = (cross - mu_phi_i.unsqueeze(2) * mu_r).clamp(-5, 5)
                      hsic_nw_ij[:, i_start:i_end, j0:j1] += hdiff.pow(2).sum(dim=-1)

              nw_resid_var_ji = nw_resid_var_ij.transpose(1, 2)
              hsic_nw_ji = hsic_nw_ij.transpose(1, 2)
              hsic_nw_ij = hsic_nw_ij.clamp(max=10).to(x_val.dtype)
              hsic_nw_ji = hsic_nw_ji.clamp(max=10).to(x_val.dtype)
              nw_resid_var_ij_out = nw_resid_var_ij.to(x_val.dtype)
              nw_resid_var_ji_out = nw_resid_var_ji.to(x_val.dtype)
              resid_skew_asym = (nw_resid_skew_ij -
                                  nw_resid_skew_ij.transpose(1, 2)).clamp(-5, 5).to(x_val.dtype)
              resid_kurt_ij_out = nw_resid_kurt_ij.to(x_val.dtype)
              resid_kurt_ji_out = nw_resid_kurt_ij.transpose(1, 2).to(x_val.dtype)

            # 18. Conditional entropy asymmetry: H(Y|X) - H(X|Y)
            Q = self.N_ENTROPY_BINS
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
              x_sorted = x_f.clone()
              x_sorted[~obs.bool()] = float('inf')
              x_sorted = x_sorted.sort(dim=1).values
              n_obs_per = obs.float().sum(dim=1).clamp(min=Q).long()
              q_fracs = torch.linspace(0, 1, Q + 1, device=x.device)
              q_idx = (q_fracs.unsqueeze(0).unsqueeze(0) *
                       (n_obs_per.unsqueeze(-1).float() - 1)).round().long()
              q_idx = q_idx.clamp(min=0, max=N - 1)
              edges = x_sorted.permute(0, 2, 1).gather(2, q_idx)
              edges[:, :, 0] -= 1; edges[:, :, -1] += 1
              obs_f32_ce = obs.float()
              # Vectorized bucketize via searchsorted: (B*D, N) vs (B*D, Q-1) in one kernel launch
              boundaries = edges[:, :, 1:-1].contiguous()  # (B, D, Q-1)
              x_flat = x_f.permute(0, 2, 1).reshape(B * D, N)  # (B*D, N)
              bnd_flat = boundaries.reshape(B * D, Q - 1)  # (B*D, Q-1)
              bins = torch.searchsorted(bnd_flat, x_flat).reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)
              one_hot = torch.nn.functional.one_hot(bins, Q).float()
              one_hot = one_hot * obs_f32_ce.unsqueeze(-1)
              x_obs_f = x_f * obs_f32_ce
              x2_obs_f = (x_f ** 2) * obs_f32_ce
              sum_y = torch.einsum('bniq,bnj->bijq', one_hot, x_obs_f)
              sum_y2 = torch.einsum('bniq,bnj->bijq', one_hot, x2_obs_f)
              jbin_count = torch.einsum('bniq,bnj->bijq', one_hot, obs_f32_ce)
              jbin_count_safe = jbin_count.clamp(min=2)
              mean_y = sum_y / jbin_count_safe
              var_y = (sum_y2 / jbin_count_safe - mean_y ** 2).clamp(min=1e-8)
              h_per_bin = 0.5 * torch.log(var_y)
              total_joint = jbin_count.sum(dim=-1, keepdim=True).clamp(min=1)
              p_bin = jbin_count / total_joint
              valid_bin = (jbin_count >= 2).float()
              cond_ent_ij = (p_bin * h_per_bin * valid_bin).sum(dim=-1)

              # 26-27. Conditional variance CV: how much does Var(Y|X=q) vary across bins?
              mean_var_y = (p_bin * var_y * valid_bin).sum(dim=-1).clamp(min=1e-8)
              var_of_var = (p_bin * (var_y - mean_var_y.unsqueeze(-1)) ** 2 * valid_bin).sum(dim=-1)
              cond_var_cv_ij = (var_of_var.sqrt() / mean_var_y).clamp(0, 5)

              # 24-25. Local R² variance: group 20 entropy bins into 5 super-bins
              Q_LR2 = 5
              grp = Q // Q_LR2  # 4 bins per super-group
              oh5 = one_hot.reshape(B, N, D, Q_LR2, grp).sum(dim=-1)  # (B, N, D, 5)
              cnt5 = jbin_count.reshape(B, D, D, Q_LR2, grp).sum(dim=-1)
              cnt5_safe = cnt5.clamp(min=3)
              valid5 = (cnt5 >= 3).float()
              sy5 = sum_y.reshape(B, D, D, Q_LR2, grp).sum(dim=-1)
              sy25 = sum_y2.reshape(B, D, D, Q_LR2, grp).sum(dim=-1)
              # Cross-products for R²: need sum(x_i * x_j) and sum(x_i²) per bin
              wxi5 = oh5 * x_obs_f.unsqueeze(-1)  # (B, N, D, 5)
              sxi5 = torch.einsum('bniq,bnj->bijq', wxi5, obs_f32_ce)
              sxi25 = torch.einsum('bniq,bnj->bijq', oh5 * x2_obs_f.unsqueeze(-1), obs_f32_ce)
              sxixj5 = torch.einsum('bniq,bnj->bijq', wxi5, x_obs_f)
              del wxi5
              my5 = sy5 / cnt5_safe
              mxi5 = sxi5 / cnt5_safe
              vy5 = (sy25 / cnt5_safe - my5 ** 2).clamp(min=1e-8)
              vxi5 = (sxi25 / cnt5_safe - mxi5 ** 2).clamp(min=1e-8)
              cov5 = sxixj5 / cnt5_safe - mxi5 * my5
              # Compute correlation first (bounded [-1,1]) then square — avoids cov²
              # overflow in float32 when pre-standardized data has extreme values
              denom5 = (vxi5 * vy5).sqrt().clamp(min=1e-4)
              corr5 = (cov5 / denom5).clamp(-1, 1)
              r2_5 = corr5 ** 2
              n_valid5 = valid5.sum(dim=-1).clamp(min=1)
              r2_mean = (r2_5 * valid5).sum(dim=-1) / n_valid5
              r2_dev = (r2_5 - r2_mean.unsqueeze(-1)) ** 2 * valid5
              local_r2_var_ij = (r2_dev.sum(dim=-1) / n_valid5).clamp(0, 5)

            cond_entropy_asym = (cond_ent_ij - cond_ent_ij.transpose(1, 2)).clamp(-5, 5).to(x_val.dtype)
            cond_var_cv_ij_out = torch.nan_to_num(cond_var_cv_ij, nan=0.0).to(x_val.dtype)
            cond_var_cv_ji_out = torch.nan_to_num(cond_var_cv_ij.transpose(1, 2), nan=0.0).to(x_val.dtype)
            local_r2_var_ij_out = torch.nan_to_num(local_r2_var_ij, nan=0.0).to(x_val.dtype)
            local_r2_var_ji_out = torch.nan_to_num(local_r2_var_ij.transpose(1, 2), nan=0.0).to(x_val.dtype)

            # 19. IGCI asymmetry -- Information-Geometric Causal Inference
            # CDF-transform then sort by x_i, compute log|dy/dx|; igci_asym = igci_ij - igci_ji
            # Janzing et al. 2012 requires uniform reference measure (empirical CDF)
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
              x_igci = x.float()
              obs_bool = obs.bool()

              # CDF transform: rank/n_obs per variable (IGCI requires uniform reference measure)
              x_igci_cdf = torch.zeros_like(x_igci)
              for j_cdf in range(D):
                  col = x_igci[:, :, j_cdf].clone()  # (B, N)
                  obs_j = obs_bool[:, :, j_cdf]
                  col[~obs_j] = float('inf')  # push unobserved to end of sort
                  sorted_idx = col.argsort(dim=1)
                  ranks = torch.zeros_like(col)
                  ranks.scatter_(1, sorted_idx, torch.arange(N, device=col.device).float().unsqueeze(0).expand(B, -1))
                  n_obs = obs_j.float().sum(dim=1, keepdim=True).clamp(min=1)
                  ranks = ranks / (n_obs - 1).clamp(min=1)  # normalize to [0, 1]
                  ranks[~obs_j] = 0.5  # neutral value for unobserved
                  x_igci_cdf[:, :, j_cdf] = ranks
              x_igci = x_igci_cdf

              igci_ij = torch.zeros(B, D, D, device=x.device)
              igci_var_ij = torch.zeros(B, D, D, device=x.device)  # IGCI slope variance

              igci_chunk = min(D, 15)
              for ci in range(0, D, igci_chunk):
                  ci_end = min(ci + igci_chunk, D)
                  n_ci = ci_end - ci

                  x_cond = x_igci[:, :, ci:ci_end].clone()
                  x_cond[~obs_bool[:, :, ci:ci_end]] = float('inf')
                  sort_idx = x_cond.argsort(dim=1)

                  x_cond_sorted = x_cond.gather(1, sort_idx)
                  obs_cond_sorted = obs_bool[:, :, ci:ci_end].gather(1, sort_idx)

                  dx = x_cond_sorted[:, 1:, :] - x_cond_sorted[:, :-1, :]
                  dx_valid = obs_cond_sorted[:, 1:, :] & obs_cond_sorted[:, :-1, :]
                  dx_abs = dx.abs().clamp(min=1e-8)
                  log_dx = torch.log(dx_abs)

                  for cj in range(0, D, igci_chunk):
                      cj_end = min(cj + igci_chunk, D)
                      n_cj = cj_end - cj

                      x_tgt = x_igci[:, :, cj:cj_end]
                      obs_tgt = obs_bool[:, :, cj:cj_end]

                      # Vectorized gather: expand sort_idx to cover all target vars at once
                      idx_exp = sort_idx.unsqueeze(-1).expand(-1, -1, -1, n_cj)  # (B, N, n_ci, n_cj)
                      x_tgt_exp = x_tgt.unsqueeze(2).expand(-1, -1, n_ci, -1)  # (B, N, n_ci, n_cj)
                      x_tgt_sorted = x_tgt_exp.gather(1, idx_exp)
                      obs_tgt_exp = obs_tgt.unsqueeze(2).expand(-1, -1, n_ci, -1)
                      obs_tgt_sorted = obs_tgt_exp.gather(1, idx_exp)

                      dy = x_tgt_sorted[:, 1:, :, :] - x_tgt_sorted[:, :-1, :, :]
                      dy_abs = dy.abs().clamp(min=1e-8)
                      log_dy = torch.log(dy_abs)

                      dy_valid = obs_tgt_sorted[:, 1:, :, :] & obs_tgt_sorted[:, :-1, :, :]
                      pair_valid = (dx_valid.unsqueeze(-1) & dy_valid).float()

                      log_ratio = (log_dy - log_dx.unsqueeze(-1)).clamp(-10, 10)
                      log_ratio = log_ratio * pair_valid

                      n_valid = pair_valid.sum(dim=1).clamp(min=1)
                      igci_mean = log_ratio.sum(dim=1) / n_valid
                      igci_mean = torch.nan_to_num(igci_mean, nan=0.0, posinf=0.0, neginf=0.0)

                      igci_ij[:, ci:ci_end, cj:cj_end] = igci_mean
                      # IGCI variance: mechanism complexity — how much log|dy/dx| varies
                      igci_sq = (log_ratio ** 2 * pair_valid).sum(dim=1) / n_valid
                      igci_var = (igci_sq - igci_mean ** 2).clamp(0, 20)
                      igci_var = torch.nan_to_num(igci_var, nan=0.0, posinf=0.0, neginf=0.0)
                      igci_var_ij[:, ci:ci_end, cj:cj_end] = igci_var

              igci_asym = (igci_ij - igci_ij.transpose(1, 2)).clamp(-10, 10).to(x_val.dtype)
            # IGCI variance outputs (from IGCI loop above)
            igci_var_ij_out = igci_var_ij.to(x_val.dtype)
            igci_var_ji_out = igci_var_ij.transpose(1, 2).to(x_val.dtype)
            # VCM outputs (from NW loop above)
            vcm_ij_out = vcm_ij.to(x_val.dtype)
            vcm_ji_out = vcm_ij.transpose(1, 2).to(x_val.dtype)
            # Stein score diagonal variance — per-variable, differenced for asymmetric feature
            stein_diag_asym = (stein_diag_var.unsqueeze(2) - stein_diag_var.unsqueeze(1)).clamp(-50, 50).to(x_val.dtype)

            # 30-31. Chatterjee Xi asymmetry — inherently asymmetric functional dependence
            # xi(X,Y) measures how well Y is determined by X using only ranks (Chatterjee 2021)
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
              chatterjee_xi = torch.zeros(B, D, D, device=x.device)
              x_f32_xi = x.float()
              obs_f32_xi = obs.float()
              xi_chunk = min(D, 15)
              for ci in range(0, D, xi_chunk):
                  ci_end = min(ci + xi_chunk, D)
                  n_ci = ci_end - ci
                  # Sort by conditioning variable x_i
                  x_cond = x_f32_xi[:, :, ci:ci_end].clone()
                  obs_cond = obs.bool()[:, :, ci:ci_end]
                  x_cond[~obs_cond] = float('inf')
                  sort_idx = x_cond.argsort(dim=1)  # (B, N, n_ci)
                  obs_sorted = obs_cond.gather(1, sort_idx)  # (B, N, n_ci)
                  n_obs_ci = obs_cond.float().sum(dim=1).clamp(min=2)  # (B, n_ci)
                  for cj in range(0, D, xi_chunk):
                      cj_end = min(cj + xi_chunk, D)
                      n_cj = cj_end - cj
                      # Get target variable ranks (use Spearman ranks already computed, re-normalized to [0, n])
                      # We need integer-like ranks for |r_{k+1} - r_k|
                      x_tgt = x_f32_xi[:, :, cj:cj_end]
                      obs_tgt = obs.bool()[:, :, cj:cj_end]
                      # Rank x_j among all observations (breaking ties with jitter already done for Spearman)
                      x_tgt_jitter = x_tgt + torch.randn_like(x_tgt) * 1e-7 * obs_tgt.float()
                      x_tgt_jitter[~obs_tgt] = float('inf')
                      tgt_sorted_idx = x_tgt_jitter.argsort(dim=1)
                      tgt_ranks = torch.zeros(B, N, n_cj, device=x.device)
                      tgt_ranks.scatter_(1, tgt_sorted_idx,
                                         torch.arange(N, device=x.device).float().view(1, N, 1).expand(B, N, n_cj))
                      # Gather target ranks in the order sorted by x_i
                      idx_exp = sort_idx.unsqueeze(-1).expand(-1, -1, -1, n_cj)  # (B, N, n_ci, n_cj)
                      tgt_ranks_exp = tgt_ranks.unsqueeze(2).expand(-1, -1, n_ci, -1)
                      r_sorted = tgt_ranks_exp.gather(1, idx_exp)  # (B, N, n_ci, n_cj)
                      # |r_{k+1} - r_k| for consecutive valid observations
                      # Must check BOTH conditioning AND target variables are observed
                      r_diff = (r_sorted[:, 1:, :, :] - r_sorted[:, :-1, :, :]).abs()
                      obs_tgt_sorted = obs_tgt.unsqueeze(2).expand(-1, -1, n_ci, -1).gather(1, idx_exp)
                      valid_cond = (obs_sorted[:, 1:, :] & obs_sorted[:, :-1, :]).unsqueeze(-1)
                      valid_tgt = obs_tgt_sorted[:, 1:, :, :] & obs_tgt_sorted[:, :-1, :, :]
                      valid_consec = (valid_cond & valid_tgt).float()
                      r_diff_sum = (r_diff * valid_consec).sum(dim=1)  # (B, n_ci, n_cj)
                      # Normalization: n_pairs * n_obs_tgt
                      # Ranks span {0,...,n_obs_tgt-1} (observed target values), differences scale with n_obs_tgt
                      # n_pairs = number of valid consecutive pairs, determines sum length
                      # Product gives correct xi=0 under independence regardless of missing data
                      n_pairs = valid_consec.sum(dim=1).clamp(min=1)  # (B, n_ci, n_cj)
                      n_obs_tgt_j = obs_tgt.float().sum(dim=1).clamp(min=2)  # (B, n_cj)
                      normalizer = (n_pairs * n_obs_tgt_j.unsqueeze(1)).clamp(min=1)  # (B, n_ci, n_cj)
                      xi_vals = (1.0 - 3.0 * r_diff_sum / normalizer).clamp(-0.5, 1.0)
                      chatterjee_xi[:, ci:ci_end, cj:cj_end] = xi_vals
            chatterjee_xi_ij = chatterjee_xi.to(x_val.dtype)
            chatterjee_xi_ji = chatterjee_xi.transpose(1, 2).to(x_val.dtype)

            # ===== NEW ANTISYMMETRIC FEATURES (20-29) =====

            # 20-21. Precision regression coefficients: β_{j←i} = -Ω[i,j]/Ω[j,j]
            # Linear partial effect of i on j, controlling for all other variables.
            # Multivariate signal — no bivariate feature captures this.
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
                prec_f = prec.float()
                diag_f = diag_prec.float()
                prec_regcoeff_ij = (-prec_f / diag_f.unsqueeze(1)).clamp(-5, 5)
                prec_regcoeff_ji = (-prec_f / diag_f.unsqueeze(2)).clamp(-5, 5)
                # 28. Precision diagonal log-ratio: log(Ω[i,i]) - log(Ω[j,j])
                log_diag = torch.log(diag_f)
                prec_diag_asym = (log_diag.unsqueeze(2) - log_diag.unsqueeze(1)).clamp(-10, 10)
                # 29. Precision row-sum ratio (off-diagonal only)
                prec_rs = (prec_f.abs().sum(dim=-1) - diag_f).clamp(min=1e-8)
                log_rs = torch.log(prec_rs)
                prec_rowsum_asym = (log_rs.unsqueeze(2) - log_rs.unsqueeze(1)).clamp(-10, 10)
            prec_regcoeff_ij = torch.nan_to_num(prec_regcoeff_ij, nan=0.0, posinf=5.0, neginf=-5.0).to(x_val.dtype)
            prec_regcoeff_ji = torch.nan_to_num(prec_regcoeff_ji, nan=0.0, posinf=5.0, neginf=-5.0).to(x_val.dtype)
            prec_diag_asym = torch.nan_to_num(prec_diag_asym, nan=0.0, posinf=10.0, neginf=-10.0).to(x_val.dtype)
            prec_rowsum_asym = torch.nan_to_num(prec_rowsum_asym, nan=0.0, posinf=10.0, neginf=-10.0).to(x_val.dtype)

            # 22-23. Nonlinearity ratio: (1 - corr²) / nw_resid_var
            # Direction head cannot compute this (corr goes only to sym head).
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
                linear_resid = (1.0 - corr.float() ** 2).clamp(min=0)
                nonlin_ratio_ij = (linear_resid / nw_resid_var_ij_out.float().clamp(min=0.01)).clamp(0, 10)
                nonlin_ratio_ji = (linear_resid / nw_resid_var_ji_out.float().clamp(min=0.01)).clamp(0, 10)
            nonlin_ratio_ij = torch.nan_to_num(nonlin_ratio_ij, nan=0.0, posinf=10.0, neginf=0.0).to(x_val.dtype)
            nonlin_ratio_ji = torch.nan_to_num(nonlin_ratio_ji, nan=0.0, posinf=10.0, neginf=0.0).to(x_val.dtype)

            # ===== STACK ALL 37 ORIGINAL CAUSAL FEATURES =====
            stats = torch.stack([
                corr,                  # 0: symmetric
                skew_prod,             # 1: symmetric
                partial_corr,          # 2: symmetric
                spearman,              # 3: symmetric
                nonlin_dep,            # 4: symmetric
                kurt_prod,             # 5: symmetric
                quad_dep,              # 6: symmetric
                nw_resid_var_ij_out,   # 7: antisymmetric
                nw_resid_var_ji_out,   # 8: antisymmetric
                skew_diff,             # 9: antisymmetric
                kurt_diff,             # 10: antisymmetric
                xi2_xj_feat,           # 11: antisymmetric
                xi_xj2_feat,           # 12: antisymmetric
                hsic_nw_ij,            # 13: antisymmetric
                hsic_nw_ji,            # 14: antisymmetric
                resid_skew_asym,       # 15: antisymmetric
                resid_kurt_ij_out,     # 16: antisymmetric
                resid_kurt_ji_out,     # 17: antisymmetric
                cond_entropy_asym,     # 18: antisymmetric
                igci_asym,             # 19: antisymmetric
                prec_regcoeff_ij,      # 20: antisymmetric (paired with 21)
                prec_regcoeff_ji,      # 21: antisymmetric
                nonlin_ratio_ij,       # 22: antisymmetric (paired with 23)
                nonlin_ratio_ji,       # 23: antisymmetric
                local_r2_var_ij_out,   # 24: antisymmetric (paired with 25)
                local_r2_var_ji_out,   # 25: antisymmetric
                cond_var_cv_ij_out,    # 26: antisymmetric (paired with 27)
                cond_var_cv_ji_out,    # 27: antisymmetric
                prec_diag_asym,        # 28: antisymmetric (direct)
                prec_rowsum_asym,      # 29: antisymmetric (direct)
                chatterjee_xi_ij,      # 30: antisymmetric (paired with 31)
                chatterjee_xi_ji,      # 31: antisymmetric
                vcm_ij_out,            # 32: antisymmetric (paired with 33)
                vcm_ji_out,            # 33: antisymmetric
                igci_var_ij_out,       # 34: antisymmetric (paired with 35)
                igci_var_ji_out,       # 35: antisymmetric
                stein_diag_asym,       # 36: antisymmetric (direct, already differenced)
            ], dim=-1)

            # ===== V-STRUCTURE (COLLIDER) EVIDENCE FEATURES (37-40) =====
            with torch.amp.autocast(device_type=x_val.device.type, enabled=False):
                corr_f = corr.float()
                corr_abs_f = corr_f.abs()
                pcorr_abs_f = partial_corr.float().abs()
                vm_f = var_mask.float()
                not_self = 1.0 - torch.eye(D, device=x.device)

                # Feature 37: collider_evidence (symmetric)
                # max_k[min(|corr(i,k)|, |corr(j,k)|) * (1-|corr(i,j)|)]
                # Chunked over i to limit memory (D^3 intermediate)
                collider_ev = torch.zeros(B, D, D, device=x.device)
                indep_ij = (1.0 - corr_abs_f).clamp(min=0)
                chunk_sz = min(D, 15)
                for i0 in range(0, D, chunk_sz):
                    i1 = min(i0 + chunk_sz, D)
                    ca_ik = corr_abs_f[:, i0:i1, :]  # (B, chunk, D)
                    # min(|corr(i,k)|, |corr(j,k)|) for all j, k
                    min_ijk = torch.minimum(
                        ca_ik.unsqueeze(2),        # (B, chunk, 1, D_k)
                        corr_abs_f.unsqueeze(1),   # (B, 1, D_j, D_k)
                    )  # (B, chunk, D_j, D_k)
                    # k-mask: k is valid, k != i, k != j
                    k_valid = vm_f.unsqueeze(1).unsqueeze(1) * not_self[i0:i1, :].unsqueeze(0).unsqueeze(2) * not_self.unsqueeze(0).unsqueeze(1)
                    best_child = (min_ijk * k_valid).max(dim=-1).values  # (B, chunk, D_j)
                    collider_ev[:, i0:i1, :] = (best_child * indep_ij[:, i0:i1, :]).clamp(0, 5)

                # Feature 38: pcorr_gap (symmetric)
                # |partial_corr(i,j)| - |corr(i,j)|  — positive means explaining-away effect
                pcorr_gap = (pcorr_abs_f - corr_abs_f).clamp(-2, 2)

                # Features 39-40: collider_child_ij/ji (asymmetric paired)
                # collider_child[i,j] = max_k[min(|corr(i,j)|, |corr(k,j)|) * (1-|corr(i,k)|)]
                # High when j looks like a child: both i and some k correlate with j, but i ⊥ k
                collider_child = torch.zeros(B, D, D, device=x.device)
                for i0 in range(0, D, chunk_sz):
                    i1 = min(i0 + chunk_sz, D)
                    r_ij = corr_abs_f[:, i0:i1, :]  # (B, chunk, D_j)
                    ca_ik = corr_abs_f[:, i0:i1, :]  # (B, chunk, D_k)
                    indep_ik = (1.0 - ca_ik).clamp(min=0)  # (B, chunk, D_k)
                    # min(|corr(i,j)|, |corr(k,j)|) for all j, k
                    min_child = torch.minimum(
                        r_ij.unsqueeze(-1),         # (B, chunk, D_j, 1)
                        corr_abs_f.unsqueeze(1),    # (B, 1, D_j, D_k)
                    )  # (B, chunk, D_j, D_k)
                    child_raw = min_child * indep_ik.unsqueeze(2)  # (B, chunk, D_j, D_k)
                    k_valid = vm_f.unsqueeze(1).unsqueeze(1) * not_self[i0:i1, :].unsqueeze(0).unsqueeze(2) * not_self.unsqueeze(0).unsqueeze(1)
                    collider_child[:, i0:i1, :] = (child_raw * k_valid).max(dim=-1).values.clamp(0, 5)

            collider_evidence = collider_ev.to(x_val.dtype)
            pcorr_gap_feat = pcorr_gap.to(x_val.dtype)
            collider_child_ij = collider_child.to(x_val.dtype)
            collider_child_ji = collider_child.transpose(1, 2).to(x_val.dtype)

            # Append V-structure features (indices 37-40) to the 37 causal features
            vstruct = torch.stack([collider_evidence, pcorr_gap_feat, collider_child_ij, collider_child_ji], dim=-1)  # (B, D, D, 4)
            stats = torch.cat([stats, vstruct], dim=-1)  # (B, D, D, 41)

            # ===== RELIABILITY METADATA FEATURES (41-44) =====
            # These are metadata about data quality, NOT causal features.
            # Never dropped by feature dropout (see Encoder stat_bias path).

            # 41. Log condition number proxy: log(max_diag_prec / min_diag_prec)
            diag_prec_rel = prec.diagonal(dim1=-2, dim2=-1)  # (B, D)
            diag_prec_valid = diag_prec_rel.abs() * var_mask + (1 - var_mask)  # padded -> 1.0
            max_diag = diag_prec_valid.max(dim=-1, keepdim=True).values  # (B, 1)
            min_diag = diag_prec_valid.min(dim=-1, keepdim=True).values.clamp(min=1e-6)  # (B, 1)
            log_cond = torch.log(max_diag / min_diag).clamp(-10, 10).unsqueeze(1).expand(-1, D, D)  # (B, D, D)

            # 42. Dimension/sample ratio: D / mean(obs_count)
            mean_n = joint_count.mean(dim=(1, 2)).clamp(min=1)  # (B,)
            d_over_n = (D_cur / mean_n).clamp(0, 1).unsqueeze(1).unsqueeze(2).expand(-1, D, D)  # (B, D, D)

            # 43. Unique ratio per variable: std/range proxy — uniform ~0.29, binary near 0
            x_range = x_val.max(dim=1).values - x_val.min(dim=1).values  # (B, D)
            x_std = x_val.std(dim=1)  # (B, D)
            unique_proxy = (x_std / x_range.clamp(min=1e-8)).clamp(0, 1)  # (B, D)
            unique_proxy = unique_proxy * var_mask  # zero for padded
            unique_i = unique_proxy.unsqueeze(2).expand(-1, -1, D)  # (B, D, D)
            unique_j = unique_proxy.unsqueeze(1).expand(-1, D, -1)  # (B, D, D)
            unique_ratio = torch.minimum(unique_i, unique_j)  # (B, D, D)

            # 44. Normalized observation count per pair
            max_possible = x_val.shape[1]  # N (samples)
            norm_obs = (joint_count / max(max_possible, 1)).clamp(0, 1)  # (B, D, D)

            # Append reliability features (indices 41-44)
            reliability = torch.stack([log_cond, d_over_n, unique_ratio, norm_obs], dim=-1)  # (B, D, D, 4)
            stats = torch.cat([stats, reliability], dim=-1)  # (B, D, D, 45)

            # Mask invalid pairs and sanitize — a single NaN in any feature
            # would poison StatAttentionBias (attention) and EdgePredictor (existence/direction)
            vm = var_mask.unsqueeze(1) * var_mask.unsqueeze(2)  # (B, D, D)
            stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
            stats = stats * vm.unsqueeze(-1)

        return stats


class TriangularEdgeRefinement(nn.Module):
    """AlphaFold2-inspired triangular pair refinement for edge features.

    Maintains a D×D×H_pair edge feature tensor and refines it through
    triangle-consistent aggregation: for each pair (i,j), aggregates
    evidence from all intermediate variables k about paths i→k→j and
    k→i, j→k. This lets the model reason about structural patterns
    (transitivity, mediation, v-structures) that per-pair prediction misses.

    Adapted from Jumper et al. (Nature 2021) Evoformer triangular
    multiplicative updates, with 1/sqrt(D) normalization for variable-count
    invariance across D=2-50. Pre-round LayerNorm on pair features prevents
    slow magnitude drift across rounds and training epochs.
    """
    def __init__(self, pair_dim, num_rounds=4):
        super().__init__()
        self.num_rounds = num_rounds
        # Pre-round LayerNorm: normalizes pair feature magnitude before
        # gate/val projections, preventing slow drift across rounds and epochs
        self.round_norms = nn.ModuleList([nn.LayerNorm(pair_dim) for _ in range(num_rounds)])
        self.rounds = nn.ModuleList()
        for _ in range(num_rounds):
            self.rounds.append(nn.ModuleDict({
                # Outgoing: aggregate i→k, k→j to update i→j
                'out_gate': nn.Linear(pair_dim, pair_dim),
                'out_val': nn.Linear(pair_dim, pair_dim),
                'out_proj': nn.Linear(pair_dim, pair_dim),
                'out_norm': nn.LayerNorm(pair_dim),
                # Incoming (fork): aggregate k→i, k→j to update i→j (common parent detection)
                'in_gate': nn.Linear(pair_dim, pair_dim),
                'in_val': nn.Linear(pair_dim, pair_dim),
                'in_proj': nn.Linear(pair_dim, pair_dim),
                'in_norm': nn.LayerNorm(pair_dim),
                # Collider: aggregate i→k, j→k to detect v-structures (i→k←j)
                'col_gate': nn.Linear(pair_dim, pair_dim),
                'col_val': nn.Linear(pair_dim, pair_dim),
                'col_proj': nn.Linear(pair_dim, pair_dim),
                'col_norm': nn.LayerNorm(pair_dim),
            }))
        # Zero-init output projections for residual stability
        for r in self.rounds:
            nn.init.zeros_(r['out_proj'].weight)
            nn.init.zeros_(r['out_proj'].bias)
            nn.init.zeros_(r['in_proj'].weight)
            nn.init.zeros_(r['in_proj'].bias)
            nn.init.zeros_(r['col_proj'].weight)
            nn.init.zeros_(r['col_proj'].bias)
        self.pair_attns = nn.ModuleList([
            nn.MultiheadAttention(pair_dim, num_heads=4, batch_first=True)
            for _ in range(num_rounds)
        ])
        self.pair_attn_norms = nn.ModuleList([
            nn.LayerNorm(pair_dim) for _ in range(num_rounds)
        ])

    @torch.compiler.disable
    def _fp32_pair_attn(self, e_normed, attn_layer, pair_kpm):
        """Pair self-attention in float32 to prevent Q@K^T overflow in bf16 at large D."""
        with torch.amp.autocast(device_type=e_normed.device.type, enabled=False):
            e_attn, _ = attn_layer(e_normed.float(), e_normed.float(), e_normed.float(), key_padding_mask=pair_kpm)
        return e_attn

    @torch.compiler.disable
    def _fp32_triangle(self, e, gate_layer, val_layer, einsum_eq, inv_sqrt_d, k_mask):
        with torch.amp.autocast(device_type=e.device.type, enabled=False):
            e_f = e.float()
            gate = (torch.tanh(gate_layer(e_f)) * 0.5 + 0.5) * k_mask
            val = val_layer(e_f)
            update = torch.einsum(einsum_eq, gate, val) * inv_sqrt_d
        return update

    def forward(self, e, var_mask):
        """e: (B, D, D, H_pair), var_mask: (B, D). Returns refined e."""
        B, D, _, H = e.shape
        vm = (var_mask.unsqueeze(1) * var_mask.unsqueeze(2))  # (B, D, D)
        diag = 1 - torch.eye(D, device=e.device)
        valid = (vm * diag).unsqueeze(-1)  # (B, D, D, 1)
        inv_sqrt_d = D ** -0.5
        # Mask for intermediate variable k in einsums: zero out padded k positions
        # Outgoing einsum bikh,bkjh: k is dim 2 of gate (bikh) → mask shape (B, 1, D, 1)
        k_mask_out = var_mask.unsqueeze(1).unsqueeze(-1).float().detach()  # detach: no grad needed, avoids compile boundary issue
        k_mask_in = var_mask.unsqueeze(-1).unsqueeze(-1).float().detach()

        # Padding mask for pair self-attention
        pair_pad = (~var_mask.bool())  # (B, D) True=padded
        # Ensure no fully-masked rows (would cause softmax NaN from -inf)
        all_masked = pair_pad.all(dim=-1, keepdim=True)  # (B, 1) — shouldn't happen but defensive
        pair_pad_safe = pair_pad.clone()
        pair_pad_safe[:, 0] = pair_pad_safe[:, 0] & ~all_masked.squeeze(-1)  # unmask first position if all masked

        for ri, (rn, r) in enumerate(zip(self.round_norms, self.rounds)):
            # Pre-round normalization: cap pair feature magnitude
            e = rn(e)

            out_update = self._fp32_triangle(e, r['out_gate'], r['out_val'], 'bikh,bkjh->bijh', inv_sqrt_d, k_mask_out)
            e = e + r['out_norm'](r['out_proj'](out_update.to(e.dtype))) * valid

            in_update = self._fp32_triangle(e, r['in_gate'], r['in_val'], 'bkih,bkjh->bijh', inv_sqrt_d, k_mask_in)
            e = e + r['in_norm'](r['in_proj'](in_update.to(e.dtype))) * valid

            # Collider: i→k and j→k both point at k — detects v-structures (i→k←j)
            col_update = self._fp32_triangle(e, r['col_gate'], r['col_val'], 'bikh,bjkh->bijh', inv_sqrt_d, k_mask_out)
            e = e + r['col_norm'](r['col_proj'](col_update.to(e.dtype))) * valid
            # Row-wise pair self-attention: each row i attends across all targets j
            B_e, D_e, _, H_e = e.shape
            e_row = e.reshape(B_e * D_e, D_e, H_e)
            e_normed = self.pair_attn_norms[ri](e_row)
            pair_kpm = pair_pad_safe.unsqueeze(1).expand(B_e, D_e, D_e).reshape(B_e * D_e, D_e)
            e_attn = self._fp32_pair_attn(e_normed, self.pair_attns[ri], pair_kpm)
            e_row = e_row + e_attn.to(e_row.dtype)
            e = e_row.reshape(B_e, D_e, D_e, H_e) * valid
            e = torch.nan_to_num(e.clamp(-5000, 5000), nan=0.0, posinf=5000.0, neginf=-5000.0)  # prevent accumulation overflow across rounds

        return e


class EdgePredictor(nn.Module):
    """Factored edge prediction: existence (symmetric) + direction (antisymmetric).

    Decomposes ``P(i -> j) = P(edge exists) * P(direction is i -> j)``:
      * Existence head consumes symmetric features
        ``[z_i + z_j, z_i * z_j, projected symmetric stats, conf_score]``.
      * Direction head consumes ``z_i - z_j`` plus encoder-gated antisymmetric
        pairwise statistics.
    An MLP gate combines ``z_i - z_j`` with all 45 raw statistics to decide
    per-pair which antisymmetric features are reliable. A fusion MLP blends
    the two heads into a single logit per ordered pair.
    """
    # Symmetric feature indices (9 features: 7 original + 2 V-structure)
    SYM_IDX = [0, 1, 2, 3, 4, 5, 6, 37, 38]
    N_SYM = 9
    # Antisymmetric paired features (differenced: f[..., a] - f[..., b])
    ASYM_PAIRS = [(7, 8), (13, 14), (11, 12), (16, 17), (20, 21), (22, 23), (24, 25), (26, 27),
                  (30, 31), (32, 33), (34, 35), (39, 40)]
    # Directly antisymmetric features (no differencing)
    ASYM_DIRECT = [9, 10, 15, 18, 19, 28, 29, 36]
    N_ASYM = 20  # 12 differenced pairs + 8 direct

    def __init__(self, hidden_dim, logit_bias_init=-2.0, num_stat_features=45):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.num_stat_features = num_stat_features
        stat_dim = hidden_dim // 4
        # Symmetric stats projection -> existence head
        self.stat_proj_sym = nn.Sequential(nn.Linear(self.N_SYM, stat_dim), nn.GELU())
        # Antisymmetric stats projection -> direction head
        self.stat_proj_asym = nn.Sequential(nn.Linear(self.N_ASYM, stat_dim), nn.GELU())
        # Encoder-conditioned feature gate: learns when each asymmetric feature is reliable
        # Input: z_i-z_j (encoder assessment of pair) + all 45 raw stats (41 causal + 4 reliability metadata)
        # Output: 20 per-feature gates in [0,1] via sigmoid
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim + num_stat_features, 256), nn.GELU(),
            nn.Linear(256, self.N_ASYM)
        )
        # Existence head: symmetric features
        exist_in = 2 * hidden_dim + stat_dim + 1  # z_i+z_j, z_i*z_j, sym_stats, conf_score
        self.exist_mlp = nn.Sequential(
            nn.Linear(exist_in, hidden_dim), nn.GELU())
        # Direction head: z_i - z_j + antisymmetric stats
        dir_in = hidden_dim + stat_dim  # z_i-z_j + projected asym_stats
        self.dir_mlp = nn.Sequential(
            nn.Linear(dir_in, hidden_dim), nn.GELU())
        # Fusion: combine existence and direction evidence
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.xavier_uniform_(self.fusion[0].weight, gain=2.0)
        nn.init.normal_(self.fusion[-1].weight, std=0.02)
        nn.init.constant_(self.fusion[-1].bias, logit_bias_init)  # sparsity prior on both MLP and tri paths

    @torch.compiler.disable
    def _fp32_predict(self, sym_feat, asym_feat, out_dtype):
        with torch.amp.autocast(device_type=sym_feat.device.type, enabled=False):
            exist_hidden = self.exist_mlp(sym_feat.float())
            dir_hidden = self.dir_mlp(asym_feat.float())
            logits = self.fusion(torch.cat([exist_hidden, dir_hidden], dim=-1)).squeeze(-1)
        return logits.to(out_dtype), exist_hidden.to(out_dtype), dir_hidden.to(out_dtype)

    def forward(self, z, pair_stats=None, asym_stats=None, conf_score=None, return_hidden=False, pre_gated_asym=None):
        z = self.ln(z)  # (B, D, H)
        D = z.shape[1]
        z_i = z.unsqueeze(2).expand(-1, -1, D, -1)  # (B, D, D, H)
        z_j = z.unsqueeze(1).expand(-1, D, -1, -1)  # (B, D, D, H)
        # Existence head: symmetric features (z_i+z_j, z_i*z_j, projected symmetric stats, conf_score)
        sym_feat = torch.cat([z_i + z_j, z_i * z_j], dim=-1)  # (B, D, D, 2H)
        if pair_stats is not None:
            # Extract symmetric features (indices 0-6, 37-38) for existence head
            sym_only = pair_stats[..., self.SYM_IDX]  # (B, D, D, N_SYM=9)
            sym_feat = torch.cat([sym_feat, self.stat_proj_sym(sym_only)], dim=-1)
        else:
            # Pad with zeros when no pair_stats provided (e.g., disable_pairwise_stats=True)
            stat_dim = self.stat_proj_sym[0].out_features  # get stat_dim from Linear layer
            sym_feat = torch.cat([sym_feat, torch.zeros(*sym_feat.shape[:-1], stat_dim, device=z.device)], dim=-1)
        if conf_score is not None:
            sym_feat = torch.cat([sym_feat, conf_score.unsqueeze(-1)], dim=-1)
        else:
            sym_feat = torch.cat([sym_feat, torch.zeros(*sym_feat.shape[:-1], 1, device=z.device)], dim=-1)
        # Direction head: z_i - z_j + gated antisymmetric stats
        # The feature gate uses the encoder's representation (z_i-z_j) + all pairwise stats
        # to decide per-pair which asymmetric features are reliable
        z_diff = z_i - z_j  # (B, D, D, H)
        if pre_gated_asym is not None:
            # Pre-computed gated features from CausalDiscoveryTransformer.forward (shared with confounder)
            dir_feat = torch.cat([z_diff, self.stat_proj_asym(pre_gated_asym)], dim=-1)
        elif asym_stats is not None and pair_stats is not None:
            gate_input = torch.cat([z_diff, pair_stats], dim=-1)
            gate = torch.sigmoid(self.feature_gate(gate_input))
            gated_asym = asym_stats * gate
            dir_feat = torch.cat([z_diff, self.stat_proj_asym(gated_asym)], dim=-1)
        elif asym_stats is not None:
            # No pair_stats but have asym_stats (shouldn't happen in practice)
            dir_feat = torch.cat([z_diff, self.stat_proj_asym(asym_stats)], dim=-1)
        else:
            # Pad with zeros when no stats provided (e.g., disable_pairwise_stats=True)
            stat_dim = self.stat_proj_asym[0].out_features
            dir_feat = torch.cat([z_diff, torch.zeros(*z_diff.shape[:-1], stat_dim, device=z.device)], dim=-1)
        # Predict existence and direction (float32 section disabled from compile)
        logits, exist_hidden, dir_hidden = self._fp32_predict(sym_feat, dir_feat, z.dtype)
        logits = logits * (1 - torch.eye(D, device=z.device))
        if return_hidden:
            hidden = torch.cat([exist_hidden, dir_hidden], dim=-1)
            return logits, hidden
        return logits


def acyclicity_spectral(logits, n_iters=10):
    """Spectral radius of sigmoid(logits) via power iteration.

    Adapted from AVICI acyclicity_spectral_log. Uses left/right power
    iteration with stop_gradient on u,v. Returns estimated spectral
    radius; should be < 1 for acyclic expected graph.
    """
    probs = torch.sigmoid(logits)
    D = probs.shape[-1]
    probs = probs * (1 - torch.eye(D, device=probs.device))

    u = torch.randn(probs.shape[0], D, device=probs.device)
    v = torch.randn(probs.shape[0], D, device=probs.device)
    for _ in range(n_iters):
        u_new = torch.bmm(u.unsqueeze(1), probs).squeeze(1)
        v_new = torch.bmm(probs, v.unsqueeze(-1)).squeeze(-1)
        u = u_new / (u_new.norm(dim=-1, keepdim=True) + 1e-8)
        v = v_new / (v_new.norm(dim=-1, keepdim=True) + 1e-8)

    u, v = u.detach(), v.detach()
    Av = torch.bmm(probs, v.unsqueeze(-1)).squeeze(-1)
    uAv = (u * Av).sum(-1)
    uv = (u * v).sum(-1)
    sr = uAv / (uv + 1e-8)
    sr = torch.nan_to_num(sr, nan=0.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)
    return sr


class ConfounderModule(nn.Module):
    """Confounder detection via learnable tokens with cross-attention.

    Tokens attend directly to variable embeddings Z (not to linear
    residuals R = X - X@W, which is meaningless for nonlinear SCMs).
    """
    def __init__(self, hidden_dim, num_tokens, num_heads,
                 num_cross_layers=2, num_sym_features=9, num_asym_features=20):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.cross_layers = nn.ModuleList()
        self.cross_norms = nn.ModuleList()
        self.cross_ffns = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.cross_layers.append(nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True))
            self.cross_norms.append(nn.LayerNorm(hidden_dim))
            self.cross_ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)))

        self.loading_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))
        # Initialize loading_net output with small random weights (not zeros!)
        # and bias=-2.0 for moderate sparsity. Zero weights would block gradient flow
        # through the hidden layer, preventing variable-specific loading learning.
        nn.init.normal_(self.loading_net[-1].weight, std=0.02)
        nn.init.constant_(self.loading_net[-1].bias, -2.0)

        # Pairwise confounding bias: direct pathway from pairwise features to C_hat
        # Symmetric features (ungated) + gated asymmetric features → confounding signal
        # Captures "high corr + low partial corr = likely confounded" directly
        self.conf_stat_proj = nn.Sequential(
            nn.Linear(num_sym_features + num_asym_features, 48), nn.GELU(),
            nn.Linear(48, 1)
        )
        nn.init.constant_(self.conf_stat_proj[-1].bias, -1.0)  # moderate confounding bias

    @torch.compiler.disable
    def _fp32_cross_attn(self, tokens, z, kpm):
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            tokens = tokens.float()
            for ca, norm, ffn in zip(self.cross_layers, self.cross_norms, self.cross_ffns):
                h, _ = ca(query=norm(tokens), key=z.float(), value=z.float(), key_padding_mask=kpm)
                tokens = tokens + h
                tokens = tokens + ffn(tokens)
                tokens = torch.nan_to_num(tokens.clamp(-5000, 5000), nan=0.0, posinf=5000.0, neginf=-5000.0)
        return tokens.to(z.dtype)

    @torch.compiler.disable
    def _fp32_loading(self, z_exp, t_exp):
        with torch.amp.autocast(device_type=z_exp.device.type, enabled=False):
            _cat_in = torch.cat([z_exp, t_exp], dim=-1).float()
            _ln_out = self.loading_net(_cat_in).squeeze(-1)
            return torch.sigmoid(_ln_out)

    @torch.compiler.disable
    def _fp32_noisy_or(self, S_i, S_j, out_dtype):
        """Noisy-OR aggregation: C[i,j] = 1 - prod_k(1 - S[i,k]*S[j,k])."""
        with torch.amp.autocast(device_type=S_i.device.type, enabled=False):
            S_i_f, S_j_f = S_i.float(), S_j.float()
            C_hat = 1 - torch.prod(1 - S_i_f * S_j_f, dim=-1)
            C_hat = (C_hat + C_hat.transpose(1, 2)) / 2
            C_hat = C_hat.clamp(1e-3, 1 - 1e-3)
            C_hat_logits = torch.logit(C_hat, eps=1e-6)
        return C_hat_logits.to(out_dtype)

    @torch.compiler.disable
    def _fp32_conf_bias(self, conf_feat):
        """Pairwise confounding bias logit from symmetric + gated asymmetric features."""
        with torch.amp.autocast(device_type=conf_feat.device.type, enabled=False):
            return self.conf_stat_proj(conf_feat.float()).squeeze(-1)

    def forward(self, z, var_mask, sym_stats=None, gated_asym_stats=None):
        B, D, H = z.shape
        K = self.num_tokens
        tokens = self.tokens.expand(B, -1, -1)
        kpm = (~var_mask.bool())
        tokens = self._fp32_cross_attn(tokens, z, kpm)
        # Loadings S[i,k]
        z_exp = z.unsqueeze(2).expand(-1, -1, K, -1)
        t_exp = tokens.unsqueeze(1).expand(-1, D, -1, -1)
        S = self._fp32_loading(z_exp, t_exp)
        # Noisy-OR: C[i,j] = 1 - prod_k(1 - S[i,k]*S[j,k])
        S_i, S_j = S.unsqueeze(2), S.unsqueeze(1)
        C_hat_logits = self._fp32_noisy_or(S_i, S_j, z.dtype)
        # Pairwise confounding bias from symmetric + gated asymmetric features
        # Soft-OR with noisy-OR: if either signal detects confounding, C is high
        if sym_stats is not None and gated_asym_stats is not None:
            conf_feat = torch.cat([sym_stats, gated_asym_stats], dim=-1)  # (B, D, D, 9+20=29)
            conf_bias_logit = self._fp32_conf_bias(conf_feat)  # (B, D, D) — raw logit
            C_hat_logits = C_hat_logits + conf_bias_logit.to(C_hat_logits.dtype)
        # Mask diagonal and padding with large negative logit
        diag_mask = torch.eye(D, device=z.device).bool()
        pad_mask = ~(var_mask.unsqueeze(1) * var_mask.unsqueeze(2)).bool()
        C_hat_logits = C_hat_logits.masked_fill(diag_mask, -30.0)
        C_hat_logits = C_hat_logits.masked_fill(pad_mask, -30.0)
        C_hat_prob = torch.sigmoid(C_hat_logits)
        return C_hat_prob, C_hat_logits, S


class CausalDiscoveryTransformer(nn.Module):
    """FoundCause model.

    Combines the alternating axis-swap transformer encoder, pairwise-statistics
    features, the factored edge predictor, triangular edge refinement, and the
    noisy-OR confounder module. Fully variable-count agnostic: no positional
    embeddings over variables.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.hidden_dim, cfg.num_encoder_layers,
                               cfg.num_heads, cfg.pma_num_queries, cfg.pma_num_heads,
                               feat_drop_asym=cfg.feat_drop_asym)
        self.pair_stats = PairwiseStatistics()
        self.edge_pred = EdgePredictor(cfg.hidden_dim, cfg.logit_bias_init,
                                        num_stat_features=PairwiseStatistics.NUM_FEATURES)
        # Triangular edge refinement over the pair-feature tensor.
        if cfg.num_tri_refine_rounds > 0:
            self.tri_proj_in = nn.Linear(cfg.hidden_dim * 2, cfg.tri_pair_dim)
            self.tri_refine = TriangularEdgeRefinement(cfg.tri_pair_dim, cfg.num_tri_refine_rounds)
            self.tri_logit = nn.Linear(cfg.tri_pair_dim, 1)
            # Learned blend between triangular and base logits.
            self.tri_alpha_exist = nn.Parameter(torch.tensor(0.0))
            self.tri_alpha_dir = nn.Parameter(torch.tensor(0.847))
            self.skel_dir_scale = nn.Parameter(torch.tensor(0.0))
            nn.init.zeros_(self.tri_logit.weight)
            nn.init.constant_(self.tri_logit.bias, cfg.logit_bias_init)
            # Learned skeleton function replacing torch.max.
            self.skel_fn = nn.Linear(2, 1, bias=True)
            with torch.no_grad():
                self.skel_fn.weight.copy_(torch.tensor([[0.5, 0.0]]))
                self.skel_fn.bias.zero_()
        else:
            self.tri_refine = None
        # Confounder module.
        self.confounder = ConfounderModule(cfg.hidden_dim, cfg.num_confounder_tokens,
                                           cfg.num_heads,
                                           num_sym_features=EdgePredictor.N_SYM,
                                           num_asym_features=EdgePredictor.N_ASYM) if cfg.predict_confounders else None
        # Parent/child role decomposition: structurally asymmetric direction signal.
        proj_dim = cfg.hidden_dim // 4
        self.parent_proj = nn.Linear(cfg.hidden_dim, proj_dim)
        self.child_proj = nn.Linear(cfg.hidden_dim, proj_dim)

    def _learned_skel(self, logits):
        """Learned symmetric skeleton extraction replacing torch.max."""
        _sym_sum = logits + logits.transpose(1, 2)
        _sym_prod = logits * logits.transpose(1, 2)
        skel_input = torch.stack([_sym_sum, _sym_prod], dim=-1)  # (B, D, D, 2)
        return self.skel_fn(skel_input).squeeze(-1)  # (B, D, D)

    def forward(self, x_val, mask, var_mask):
        # Pairwise statistics (skip if explicitly disabled via config).
        if self.cfg.disable_pairwise_stats:
            stats = None
            asym_stats = None
        else:
            stats = self.pair_stats(x_val, mask, var_mask)
            # Compute antisymmetric stats: difference paired features, collect direct features.
            asym_parts = []
            for a, b in EdgePredictor.ASYM_PAIRS:
                asym_parts.append(stats[..., a] - stats[..., b])
            for idx in EdgePredictor.ASYM_DIRECT:
                asym_parts.append(stats[..., idx])
            asym_stats = torch.stack(asym_parts, dim=-1)  # (B, D, D, N_ASYM=20)
        # Encode: variable embeddings via alternating attention + PMA pooling.
        z, pma_div = self.encoder(x_val, mask, var_mask, pair_stats=stats)
        # Compute gated asymmetric features (shared by confounder and edge predictor).
        gated_asym = None
        if asym_stats is not None and stats is not None:
            D_dim = z.shape[1]
            z_normed = self.edge_pred.ln(z)
            z_i_gate = z_normed.unsqueeze(2).expand(-1, -1, D_dim, -1)
            z_j_gate = z_normed.unsqueeze(1).expand(-1, D_dim, -1, -1)
            # Float32 upcast for the gate MLP (wide fan-in would overflow bf16).
            with torch.amp.autocast(device_type=z.device.type, enabled=False):
                gate_input = torch.cat([z_i_gate - z_j_gate, stats], dim=-1).float()
                gate = torch.sigmoid(self.edge_pred.feature_gate(gate_input)).to(z.dtype)
            gated_asym = asym_stats * gate
        _gate_bounds = torch.tensor(0.0, device=z.device)
        # Extract symmetric features for confounder (ungated).
        sym_stats = stats[..., EdgePredictor.SYM_IDX] if stats is not None else None
        # Compute confounding first so edge predictor can use it.
        if self.confounder is not None:
            C_hat, C_hat_logits, S = self.confounder(z, var_mask,
                                                      sym_stats=sym_stats, gated_asym_stats=gated_asym)
            conf_score = C_hat.detach()  # prevent FP penalty gradient from suppressing confounder
        else:
            conf_score = None
            C_hat = C_hat_logits = torch.zeros(z.shape[0], z.shape[1], z.shape[1], device=z.device)
            S = torch.zeros(z.shape[0], z.shape[1], 1, device=z.device)
        if self.tri_refine is not None:
            base_logits, hidden = self.edge_pred(z, pair_stats=stats, asym_stats=asym_stats, conf_score=conf_score, return_hidden=True, pre_gated_asym=gated_asym)
            base_logits = base_logits.clamp(-15, 15)
            e = self.tri_proj_in(hidden)
            e = torch.nan_to_num(e.clamp(-5000, 5000), nan=0.0, posinf=5000.0, neginf=-5000.0)
            e = self.tri_refine(e, var_mask)
            tri_logits = self.tri_logit(e).squeeze(-1).clamp(-15, 15)
            D_dim = tri_logits.shape[-1]
            tri_logits = tri_logits * (1 - torch.eye(D_dim, device=z.device))
            # Asymmetric blending: triangular path dominates direction (V-structure reasoning)
            # MLP path dominates existence (distributional features useful for edge detection)
            skel_base = self._learned_skel(base_logits)
            skel_tri = self._learned_skel(tri_logits)
            dir_base = base_logits - base_logits.transpose(1, 2)
            dir_tri = tri_logits - tri_logits.transpose(1, 2)

            alpha_e = torch.sigmoid(self.tri_alpha_exist)
            alpha_d = torch.sigmoid(self.tri_alpha_dir)
            skel_blend = alpha_e * skel_tri + (1 - alpha_e) * skel_base
            dir_blend = alpha_d * dir_tri + (1 - alpha_d) * dir_base

            # Combine blended skeleton + direction into directed logits
            # _scale in [0.3, 0.7] — learnable replacement for fixed /2.0
            _scale = torch.sigmoid(self.skel_dir_scale).clamp(0.3, 0.7)
            edge_logits = (skel_blend + dir_blend) * _scale
        else:
            edge_logits = self.edge_pred(z, pair_stats=stats, asym_stats=asym_stats, conf_score=conf_score, pre_gated_asym=gated_asym)
        edge_logits = edge_logits.clamp(-15, 15)
        # Parent/child role decomposition: dot(V_i, U_j) / sqrt(proj_dim)
        D_dim = z.shape[1]
        V = self.parent_proj(z)  # (B, D, proj_dim) — parent role
        U = self.child_proj(z)   # (B, D, proj_dim) — child role
        proj_dim = V.shape[-1]
        pc_score = torch.bmm(V, U.transpose(1, 2)) / (proj_dim ** 0.5)  # (B, D, D)
        pc_score = pc_score * (1 - torch.eye(D_dim, device=z.device))
        pc_score = torch.nan_to_num(pc_score.clamp(-100, 100), nan=0.0, posinf=100.0, neginf=-100.0)
        edge_logits = edge_logits + 0.1 * pc_score  # blend with small weight

        edge_logits = edge_logits.clamp(-15, 15)
        edge_probs = torch.sigmoid(edge_logits)
        acyc = acyclicity_spectral(edge_logits, self.cfg.acyclicity_power_iters) if self.cfg.lambda_acyc > 0 else torch.zeros(edge_logits.shape[0], device=edge_logits.device)
        vm = var_mask.unsqueeze(1) * var_mask.unsqueeze(2)
        diag = 1 - torch.eye(edge_probs.shape[-1], device=edge_probs.device)
        return {'edge_logits': edge_logits, 'D': edge_probs * vm * diag,
                'C': C_hat, 'C_logits': C_hat_logits, 'S': S, 'acyclicity': acyc,
                'pma_div': pma_div, 'gate_bounds': _gate_bounds,
                'skel_logit': self._learned_skel(edge_logits) if self.tri_refine is not None else None}


def _gmm_threshold(D_pred, d):
    """Fit 2-component GMM to logit distribution, find adaptive threshold.

    Uses skeleton scores (max of each directed pair) in logit space for
    better separation between edge/non-edge clusters. Falls back to 0.5
    if too few pairs or components are not well-separated.
    """
    from sklearn.mixture import GaussianMixture
    # Collect unique skeleton scores (upper triangle)
    skel_logits = []
    for i in range(d):
        for j in range(i + 1, d):
            s = max(D_pred[i, j], D_pred[j, i])
            s = np.clip(s, 1e-6, 1 - 1e-6)
            skel_logits.append(np.log(s / (1 - s)))
    skel_logits = np.array(skel_logits)

    if len(skel_logits) < 4:  # too few pairs for GMM
        return 0.5

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100)
    try:
        gmm.fit(skel_logits.reshape(-1, 1))
    except Exception:
        return 0.5  # GMM convergence failure — fall back to default

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())

    # Check if components are well-separated
    idx_low = 0 if means[0] < means[1] else 1
    idx_high = 1 - idx_low
    separation = abs(means[idx_high] - means[idx_low])

    if separation < 1.0:  # not well-separated, fall back
        return 0.5

    # Find crossing point (weighted midpoint by stds)
    mu_low, mu_high = means[idx_low], means[idx_high]
    s_low, s_high = stds[idx_low], stds[idx_high]
    cross_logit = (mu_low * s_high + mu_high * s_low) / (s_low + s_high)

    # Convert back to probability
    threshold = 1.0 / (1.0 + np.exp(-cross_logit))
    # Clamp to reasonable range
    threshold = np.clip(threshold, 0.1, 0.9)
    return float(threshold)


def predict(model, X, device, temperature=1.0, n_runs=10, max_samples=5000,
            min_agreement=0.5, threshold=None, optimizer=None):
    """Predict causal graph from observational data.

    Uses torch.inference_mode() for gradient tracking. Callers should set
    model.eval() before calling to disable feature/variable dropout.

    Note: Not thread-safe. Do not call from multiple threads simultaneously.

    Args:
        model: trained CausalDiscoveryTransformer (can be DDP-wrapped)
        X: numpy array of shape (n_samples, d_variables)
        device: torch device
        temperature: calibration temperature (default 1.0)
        n_runs: number of permutation-averaged forward passes (default 10)
        max_samples: max observations to use (default 5000)
        min_agreement: fraction of runs that must agree on an edge (default 0.5)
        threshold: fixed threshold override. If None, uses adaptive GMM threshold.
        optimizer: Schedule-Free optimizer (needs .eval()/.train() for correct weights).

    Returns:
        dict with keys:
            'adjacency': binary numpy array (d, d) -- predicted DAG
            'probabilities': float numpy array (d, d) -- edge probabilities
            'threshold': float -- threshold used
            'agreement': float numpy array (d, d) -- per-edge agreement rate across runs
    """
    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model
    if optimizer is not None:
        optimizer.eval()
    use_amp = device.type == 'cuda'

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (samples, variables), got shape {X.shape}")
    n_orig, d = X.shape
    if d < 2:
        import warnings
        warnings.warn(f"Need >= 2 variables for causal discovery, got d={d}")
        return {'adjacency': np.zeros((d, d)), 'probabilities': np.zeros((d, d)),
                'threshold': 0.5, 'agreement': np.zeros((d, d))}
    if n_orig < 2:
        import warnings
        warnings.warn(f"Need >= 2 samples, got n={n_orig}")
        return {'adjacency': np.zeros((d, d)), 'probabilities': np.zeros((d, d)),
                'threshold': 0.5, 'agreement': np.zeros((d, d))}
    if d > 51:
        import warnings
        warnings.warn(f"Model trained on d<=51, got d={d}. Results may be unreliable and slow.")

    # Subsample rows if n > max_samples
    if n_orig > max_samples:
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(n_orig, max_samples, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    # K-runs loop with column permutation for robustness
    D_accum = np.zeros((d, d))
    D_runs_raw = []  # list of (d, d) probability arrays for post-hoc agreement

    with torch.inference_mode():
        for run_i in range(n_runs):
            rng_eval = np.random.RandomState(42 + run_i)
            # Random column permutation
            col_perm = rng_eval.permutation(d)
            X_perm = X_use[:, col_perm]

            # Handle NaN in input: convert to mask (real-world data may have missing values)
            nan_mask = np.isnan(X_perm)
            if nan_mask.any():
                X_perm = np.nan_to_num(X_perm, nan=0.0)

            # Prepare tensors: model expects (B, N, D) value + (B, N, D) mask + (B, D) var_mask
            X_t = torch.tensor(X_perm).unsqueeze(0).to(device)
            M_t = torch.tensor(~nan_mask if nan_mask.any() else np.ones_like(X_perm),
                               dtype=torch.float32).unsqueeze(0).to(device)
            vm_t = torch.ones(1, d, device=device)

            # Forward pass
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = raw_model(X_t, M_t, vm_t)

            # Temperature-scale the logits
            D_run = torch.sigmoid(out['edge_logits'][0, :d, :d] / temperature).cpu().numpy()

            # Un-permute back to original variable ordering
            inv_perm = np.argsort(col_perm)
            D_unperm = D_run[np.ix_(inv_perm, inv_perm)]
            D_accum += D_unperm
            D_runs_raw.append(D_unperm)

        # Average probabilities
        D_pred = D_accum / n_runs

        # Determine threshold: GMM adaptive or fixed override
        if threshold is None:
            threshold_used = _gmm_threshold(D_pred, d)
        else:
            threshold_used = float(threshold)

        # Compute per-run agreement using the actual threshold (not hardcoded 0.5)
        D_runs_bin = []
        for D_run_prob in D_runs_raw:
            D_run_bin = np.zeros((d, d))
            for i in range(d):
                for j in range(i + 1, d):
                    s = max(D_run_prob[i, j], D_run_prob[j, i])
                    if s > threshold_used:
                        if D_run_prob[i, j] >= D_run_prob[j, i]:
                            D_run_bin[i, j] = 1.0
                        else:
                            D_run_bin[j, i] = 1.0
            D_runs_bin.append(D_run_bin)
        agreement = np.mean(D_runs_bin, axis=0)  # (d, d), values in [0, 1]

        # Two-stage edge recovery: existence by max, direction by argmax
        D_pred_bin = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                skel_score = max(D_pred[i, j], D_pred[j, i])
                if skel_score > threshold_used:
                    if D_pred[i, j] >= D_pred[j, i]:
                        D_pred_bin[i, j] = 1.0
                    else:
                        D_pred_bin[j, i] = 1.0

        # Self-consistency pruning: remove edges with low agreement
        for i in range(d):
            for j in range(d):
                if D_pred_bin[i, j] > 0 and agreement[i, j] < min_agreement:
                    D_pred_bin[i, j] = 0

    if optimizer is not None:
        optimizer.train()

    return {
        'adjacency': D_pred_bin,
        'probabilities': D_pred,
        'threshold': threshold_used,
        'agreement': agreement,
    }
