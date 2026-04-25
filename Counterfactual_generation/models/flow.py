"""
Latent Transport Models for Age-Conditional Counterfactual Generation.

Two architectures:
- FlowMLP: Flow matching velocity field v(z_t, t | c), requires ODE integration
- TransportMLP: Direct displacement f(z_src | c), single forward pass

Both use the same building blocks (SinusoidalEmbedding, AdaLN, k-NN coupling).
TransportMLP is preferred — it avoids the velocity averaging problem inherent
in flow matching with diverse k-NN couplings.
"""

import torch
import torch.nn as nn
import numpy as np


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class AdaLNBlock(nn.Module):
    """MLP block with Adaptive Layer Norm conditioning."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)  # scale + shift
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.linear(x)
        h = self.act(h)
        normed = self.norm(h)
        style = self.cond_proj(cond)
        scale, shift = style.chunk(2, dim=-1)
        return normed * (1 + scale) + shift


class ConcatBlock(nn.Module):
    """Standard MLP block (for vanilla concat conditioning)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, cond=None):
        h = self.linear(x)
        h = self.act(h)
        return self.norm(h)


class FlowMLP(nn.Module):
    """
    Flow Matching MLP with configurable components.
    
    Args:
        use_adaln: Use AdaLN conditioning (True) or concat (False)
        cfg_dropout: CFG dropout rate (0 to disable CFG)
        zero_init: Initialize output head to zero
    """
    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=6,
                 time_dim=64, cond_dim=1,
                 use_adaln=True, cfg_dropout=0.1, zero_init=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_adaln = use_adaln
        self.cfg_dropout = cfg_dropout
        self.cond_dim = cond_dim

        # Embeddings
        self.time_emb = SinusoidalEmbedding(time_dim)

        # Condition embedding: sinusoidal per dimension + learned MLP
        # Each condition scalar gets its own sinusoidal features (like time),
        # then they're combined through a nonlinear projection.
        self.cond_sinusoidal = SinusoidalEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim * cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        if use_adaln:
            # Concat+AdaLN: condition concatenated at input for direct z×c
            # interaction, PLUS AdaLN modulation at every layer.
            # Pure AdaLN (no concat) limits first-layer interaction — the
            # model can't compute z-and-condition-dependent features until
            # deeper layers.  For data-to-data transport this matters because
            # the velocity depends on BOTH position z and the age pair.
            self.input_proj = nn.Linear(latent_dim + time_dim * 2, hidden_dim)
            self.blocks = nn.ModuleList([
                AdaLNBlock(hidden_dim, time_dim * 2) for _ in range(num_layers)
            ])
        else:
            # Concat-only: condition concatenated to input, no modulation
            self.input_proj = nn.Linear(latent_dim + time_dim * 2, hidden_dim)
            self.blocks = nn.ModuleList([
                ConcatBlock(hidden_dim) for _ in range(num_layers)
            ])
        
        # Output projection
        self.head = nn.Linear(hidden_dim, latent_dim)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, z, t, c, train_cfg=True):
        """
        Predict velocity v(z, t | c).
        """
        t_emb = self.time_emb(t)
        
        if c.dim() == 1:
            c = c.unsqueeze(1)

        # CFG dropout during training (use -1.0 as null token, not 0.0,
        # since age_norm=0.0 is a valid condition for the youngest age)
        if self.training and train_cfg and self.cfg_dropout > 0:
            mask = torch.rand(c.size(0), 1, device=c.device) > self.cfg_dropout
            null_c = torch.full_like(c, -1.0)
            c = torch.where(mask, c, null_c)

        # Sinusoidal embedding per condition dimension, then learned projection
        c_parts = [self.cond_sinusoidal(c[:, i]) for i in range(self.cond_dim)]
        c_emb = self.cond_proj(torch.cat(c_parts, dim=1))
        cond = torch.cat([t_emb, c_emb], dim=1)
        
        if self.use_adaln:
            # Concat at input (direct z×c interaction) + AdaLN at every layer
            h = self.input_proj(torch.cat([z, cond], dim=1))
            for block in self.blocks:
                h = h + block(h, cond)  # residual + AdaLN modulation
        else:
            # Concat-only mode
            h = self.input_proj(torch.cat([z, cond], dim=1))
            for block in self.blocks:
                h = h + block(h)  # residual connection
            
        return self.head(h)

    # ========================== Inference Methods ==========================
    
    @torch.inference_mode()
    def counterfactual(self, z_source, c_source, c_target, 
                       alpha=0.2, steps=50, cfg_scale=4.0, solver='heun'):
        """
        Generate counterfactual latent.
        
        Args:
            alpha: Identity mixing (0=locked, 0.3=balanced, 1=random)
            solver: 'euler' (1st order) or 'heun' (2nd order)
        """
        # A. INVERT: Data → Noise
        z_noise = self._integrate(
            z_source, c_source, steps, direction='backward', 
            cfg_scale=1.0, solver=solver
        )
        
        # B. ALPHA MIX: Break identity lock
        if alpha > 0:
            fresh = torch.randn_like(z_noise)
            z_mixed = (1 - alpha) * z_noise + alpha * fresh
            std_old = z_noise.std(dim=-1, keepdim=True)
            std_new = z_mixed.std(dim=-1, keepdim=True) + 1e-8
            z_noise = z_mixed * (std_old / std_new)

        # C. GENERATE: Noise → Data
        z_counterfactual = self._integrate(
            z_noise, c_target, steps, direction='forward', 
            cfg_scale=cfg_scale, solver=solver
        )
        
        return z_counterfactual

    @torch.inference_mode()
    def _integrate(self, z, c, steps, direction='forward', cfg_scale=1.0, solver='heun'):
        """ODE integration with configurable solver."""
        dt = 1.0 / steps
        t_start = 0.0 if direction == 'forward' else 1.0
        if direction == 'backward':
            dt = -dt

        def get_velocity(z_in, t_val):
            t_tensor = torch.full((z.size(0),), t_val, device=z.device)
            
            if cfg_scale == 1.0 or self.cfg_dropout == 0:
                return self(z_in, t_tensor, c, train_cfg=False)
            
            # CFG: v_uncond + scale * (v_cond - v_uncond)
            v_cond = self(z_in, t_tensor, c, train_cfg=False)
            v_uncond = self(z_in, t_tensor, torch.full_like(c, -1.0), train_cfg=False)
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        for i in range(steps):
            t = t_start + i * dt
            v1 = get_velocity(z, t)
            
            if solver == 'heun':
                # Heun (2nd order)
                z_guess = z + v1 * dt
                v2 = get_velocity(z_guess, t + dt)
                z = z + (dt / 2) * (v1 + v2)
            else:
                # Euler (1st order)
                z = z + v1 * dt
            
        return z
    
    def invert(self, z_data, c, steps=50, cfg_scale=1.0, solver='heun'):
        """Data → Noise."""
        return self._integrate(z_data, c, steps, 'backward', cfg_scale, solver)

    def generate(self, z_noise, c, steps=50, cfg_scale=1.0, solver='heun',
                 sample=False):
        """Noise → Data."""
        return self._integrate(z_noise, c, steps, 'forward', cfg_scale, solver)


class TransportMLP(nn.Module):
    """
    Direct displacement predictor: z_cf = z_src + f(z_src | age_src, age_tgt).

    Unlike FlowMLP which learns a velocity field at interpolated points z_t
    (causing velocity averaging from crossing trajectories), this predicts
    the total displacement directly from the real source point.

    Identity is preserved structurally through the residual skip connection
    (z_src + delta). The model only controls the age-dependent change.

    Args:
        latent_dim: Dimension of the latent space
        hidden_dim: Hidden layer width
        num_layers: Number of AdaLN blocks
        cond_dim: Number of condition scalars (2 = age_src + age_tgt)
        embed_dim: Dimension of sinusoidal embeddings per condition scalar
    """
    def __init__(self, latent_dim=128, hidden_dim=512, num_layers=6,
                 cond_dim=2, embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Sinusoidal embedding per condition dimension + learned projection
        self.cond_sinusoidal = SinusoidalEmbedding(embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim * cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Concat at input: z_src and condition embedding
        cond_total = embed_dim  # after projection
        self.input_proj = nn.Linear(latent_dim + cond_total, hidden_dim)

        # AdaLN blocks with residual connections
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, cond_total) for _ in range(num_layers)
        ])

        # Output: displacement delta, zero-initialized for stable start
        # (delta ≈ 0 at init means z_cf ≈ z_src, i.e. identity transport)
        self.head = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_src, c):
        """
        Predict displacement delta = z_tgt - z_src.

        Args:
            z_src: (B, latent_dim) source latents
            c: (B, cond_dim) condition [age_src_norm, age_tgt_norm]

        Returns:
            delta: (B, latent_dim) predicted displacement
        """
        if c.dim() == 1:
            c = c.unsqueeze(1)

        # Sinusoidal embedding per condition dimension
        c_parts = [self.cond_sinusoidal(c[:, i]) for i in range(self.cond_dim)]
        cond = self.cond_proj(torch.cat(c_parts, dim=1))

        # Concat at input + AdaLN at every layer
        h = self.input_proj(torch.cat([z_src, cond], dim=1))
        for block in self.blocks:
            h = h + block(h, cond)

        return self.head(h)

    # ========================== Inference Methods ==========================
    # API-compatible with FlowMLP so visualization/evaluation code works unchanged.

    @torch.inference_mode()
    def generate(self, z_src, c, steps=None, cfg_scale=None, solver=None,
                 sample=False):
        """Direct transport: z_cf = z_src + f(z_src, c). Single forward pass."""
        return z_src + self(z_src, c)


class HeteroscedasticTransportMLP(TransportMLP):
    """TransportMLP that predicts both displacement mean AND per-dimension variance.

    Instead of MSE (which regresses to the conditional mean and collapses
    diversity), this model is trained with Gaussian NLL loss:

        L = 0.5 * mean(log_var + (delta_target - delta_mean)^2 / exp(log_var))

    At gap=0, OT coupling gives noisy targets (different vocalizations paired
    together, delta_target ≈ 7 in random directions). With MSE, the model is
    forced toward the nonzero average. With NLL, the model can learn:
        delta_mean ≈ 0,  delta_var ≈ large
    because the loss doesn't penalize mean=0 as long as variance explains
    the scattered targets.

    Inference modes:
        - Deterministic (for trajectory variance): z_cf = z_src + delta_mean
          The mean prediction is identity-dependent: small for calls (age-
          invariant), large for songs (age-dependent). Use this for the
          disentanglement metric.
        - Sampled (for FAD evaluation): z_cf = z_src + delta_mean + sqrt(var)*noise
          Restores the diversity that MSE/NLL's mean prediction loses, giving
          variance ratio ≈ 1.0 without artificial rescaling.

    Args:
        latent_dim, hidden_dim, num_layers, cond_dim, embed_dim: same as TransportMLP
        log_var_clamp: clamp range for log_var to prevent numerical instability
    """
    def __init__(self, latent_dim=128, hidden_dim=512, num_layers=6,
                 cond_dim=2, embed_dim=64, log_var_clamp=6.0):
        super().__init__(latent_dim, hidden_dim, num_layers, cond_dim, embed_dim)
        self.log_var_clamp = log_var_clamp

        # Variance head: shares backbone, separate output projection
        # Initialize to log_var ≈ 0 (var = 1) as a neutral starting point
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def forward(self, z_src, c):
        """Predict displacement mean and log-variance.

        Args:
            z_src: (B, latent_dim) source latents
            c: (B, cond_dim) condition [age_src_norm, age_tgt_norm]

        Returns:
            delta_mean: (B, latent_dim) predicted mean displacement
            delta_log_var: (B, latent_dim) predicted log-variance per dimension
        """
        if c.dim() == 1:
            c = c.unsqueeze(1)

        # Shared backbone (same as TransportMLP.forward but we need h)
        c_parts = [self.cond_sinusoidal(c[:, i]) for i in range(self.cond_dim)]
        cond = self.cond_proj(torch.cat(c_parts, dim=1))
        h = self.input_proj(torch.cat([z_src, cond], dim=1))
        for block in self.blocks:
            h = h + block(h, cond)

        # Two heads from shared representation
        delta_mean = self.head(h)  # zero-initialized (identity at init)
        delta_log_var = self.log_var_head(h)
        delta_log_var = delta_log_var.clamp(-self.log_var_clamp, self.log_var_clamp)

        return delta_mean, delta_log_var

    @torch.inference_mode()
    def generate(self, z_src, c, steps=None, cfg_scale=None, solver=None,
                 sample=False, sample_temp=1.0):
        """Direct transport with optional variance sampling.

        Args:
            sample: If False (default), deterministic: z_cf = z_src + delta_mean.
                    If True, stochastic: z_cf = z_src + delta_mean + std * noise.
            sample_temp: Temperature for sampling noise (0=deterministic, 1=full).
        """
        delta_mean, delta_log_var = self(z_src, c)
        if sample:
            std = (0.5 * delta_log_var).exp() * sample_temp
            noise = torch.randn_like(std)
            return z_src + delta_mean + std * noise
        return z_src + delta_mean


class TransportMLPv2(TransportMLP):
    """TransportMLP with gap-aware gating and optional gap-prediction head.

    Improvements over v1:
    - Gap gating: delta = α(|gap|) ⊙ f(z_src, c), where α is a per-dimension
      scale vector from a small MLP. Initialized so α(0) ≈ 0, enforcing
      near-identity transport when source age ≈ target age.
    - Gap-prediction head: auxiliary head that predicts |age_gap| from the
      displacement vector, forcing displacement to encode gap magnitude info.

    Both features are independently toggleable.

    Args:
        gap_gate: Enable gap-aware gating (default True)
        gap_head: Enable gap-prediction auxiliary head (default True)
        ... (remaining args same as TransportMLP)
    """
    def __init__(self, latent_dim=128, hidden_dim=512, num_layers=6,
                 cond_dim=2, embed_dim=64,
                 gap_gate=True, gap_head=True):
        super().__init__(latent_dim, hidden_dim, num_layers,
                         cond_dim, embed_dim)
        self.use_gap_gate = gap_gate
        self.use_gap_head = gap_head

        if gap_gate:
            # MLP: |gap| (scalar) → per-dimension scale vector (latent_dim)
            # Initialized so α(0) ≈ 0: zero bias on final layer
            self.gap_gate_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, latent_dim),
                nn.Sigmoid(),  # output in [0, 1]
            )
            # Initialize final layer bias to -3 so sigmoid(-3) ≈ 0.05 at gap=0
            # (near-zero gating when gap=0)
            nn.init.zeros_(self.gap_gate_mlp[-2].weight)
            nn.init.constant_(self.gap_gate_mlp[-2].bias, -3.0)

        if gap_head:
            # Auxiliary head: displacement → predicted |gap|
            self.gap_predictor = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

    def forward(self, z_src, c):
        """
        Predict displacement delta = α(|gap|) ⊙ f(z_src, c).

        Returns:
            delta: (B, latent_dim) gated displacement
            gap_pred: (B, 1) predicted |gap| (or None if gap_head disabled)
        """
        # Get raw displacement from parent
        delta_raw = super().forward(z_src, c)

        # Apply gap gating
        if self.use_gap_gate:
            if c.dim() == 1:
                c_2d = c.unsqueeze(1)
            else:
                c_2d = c
            # |gap| = |age_tgt - age_src| (both are normalized ages)
            gap = (c_2d[:, 1] - c_2d[:, 0]).abs().unsqueeze(1)  # (B, 1)
            alpha = self.gap_gate_mlp(gap)  # (B, latent_dim)
            delta = delta_raw * alpha
        else:
            delta = delta_raw

        # Gap prediction head
        gap_pred = None
        if self.use_gap_head:
            gap_pred = self.gap_predictor(delta)  # (B, 1)

        return delta, gap_pred

    @torch.inference_mode()
    def generate(self, z_src, c, steps=None, cfg_scale=None, solver=None,
                 sample=False):
        """Direct transport: z_cf = z_src + gated_delta. Single forward pass."""
        delta, _ = self(z_src, c)
        return z_src + delta


# ========================== Factory ==========================

def load_transport_model(config, device, weights_path=None):
    """Instantiate TransportMLP/v2 or FlowMLP from a saved config dict.

    Args:
        config: dict with keys 'arch', 'latent_dim', 'hidden_dim', 'num_blocks',
                and optionally 'cond_dim', 'cfg_dropout', 'model_version'.
        device: torch device.
        weights_path: optional Path to .pt state dict. If provided, loads weights
                      and sets model to eval mode.

    Returns:
        model on device (eval mode if weights loaded, train mode otherwise).
    """
    arch = config.get('arch', 'direct')
    version = config.get('model_version', 1)

    if arch == 'direct':
        if version >= 3:
            model = HeteroscedasticTransportMLP(
                latent_dim=config['latent_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_blocks'],
                cond_dim=config.get('cond_dim', 2),
                embed_dim=64,
            ).to(device)
        elif version >= 2:
            model = TransportMLPv2(
                latent_dim=config['latent_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_blocks'],
                cond_dim=config.get('cond_dim', 2),
                embed_dim=64,
                gap_gate=config.get('gap_gate', True),
                gap_head=config.get('gap_head', True),
            ).to(device)
        else:
            model = TransportMLP(
                latent_dim=config['latent_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_blocks'],
                cond_dim=config.get('cond_dim', 2),
                embed_dim=64,
            ).to(device)
    else:
        model = FlowMLP(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_blocks'],
            time_dim=64,
            cond_dim=config.get('cond_dim', 2),
            use_adaln=True,
            cfg_dropout=config.get('cfg_dropout', 0.0),
            zero_init=True,
        ).to(device)

    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

    return model
