# Copyright (c) 2026 PsiQuantum Corp.
# Licensed under the Academic Software License v1.0 (ASL); see LICENSE and NOTICE.
"""Orbital-specific readout, long-range, and projection blocks for OrbACE."""

import math
from typing import Optional

import torch
import torch.nn as nn

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3
from e3nn.o3 import Linear


class OrbitalReadoutBlock(nn.Module):
    """Non-linear readout: per-orbital hidden features -> scalar energy."""

    def __init__(self, irreps_in: o3.Irreps, hidden_dim: int):
        super().__init__()
        n_scalars_in = irreps_in.count(o3.Irrep(0, 1))
        self.linear_1 = nn.Linear(n_scalars_in, hidden_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear_1(node_feats))
        return self.linear_2(x).squeeze(-1)


class LinearOrbitalReadoutBlock(nn.Module):
    """Linear readout for intermediate layers."""

    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("1x0e"))

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        return self.linear(node_feats).squeeze(-1)


class ScaleShiftBlock(nn.Module):
    """Per-node scale and shift, applied BEFORE summing to graph energy.

    node_energy_out = scale * node_energy_in + shift
    Then total_energy = sum(node_energy_out).

    This means shift contributes N_nodes * shift to the total energy.
    """

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift


class LongRangeBlock(nn.Module):
    """Long-range electrostatic interaction between orbital sites.

    Each orbital gets a learned charge q_i from its hidden features:
        E_qq = (1/2) sum_{i!=j} q_i * q_j * erfc(alpha * r_ij) / r_ij

    When ``use_multipoles`` is on, each orbital additionally carries an
    equivariant dipole mu_i = s_i * p_i, where s_i is a learned invariant scalar
    (read from the hidden features) and p_i is the frozen HF orbital
    polarization vector (lab frame, detached). This adds the screened
    charge-dipole and dipole-dipole terms:
        E_qd = (1/2) sum_{i!=j} (q_j (mu_i . r_ij) - q_i (mu_j . r_ij)) B1
        E_dd = (1/2) sum_{i!=j} [ (mu_i . mu_j) B1 - (mu_i . r_ij)(mu_j . r_ij) B2 ]
    with r_ij = r_i - r_j and the Ewald-damped interaction functions
        B0 = erfc(alpha r)/r,  g = (2 alpha/sqrt(pi)) exp(-alpha^2 r^2),
        B1 = (B0 + g)/r^2,     B2 = (3 B1 + 2 alpha^2 g)/r^2.
    In the bare limit (alpha->0) these reduce to 1/r, 1/r^3, 3/r^5, i.e. the
    standard multipole interaction tensors. Forces are unaffected by the frozen
    direction because p_i is detached (positions still drive r_ij).
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 0.2,
        lr_cutoff: float = 15.0,
        ewald_dl: float = 2.0,
        use_multipoles: bool = False,
        init_readout_std: float = 1.0e-3,
    ):
        super().__init__()
        self.charge_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Small random, NOT zero, and both failure modes are real.
        #
        # Default Kaiming init gives charges O(sqrt(hidden_dim)) which, against
        # an O(N^2) Coulomb sum, reach >1e7 eV on realistic batches and send the
        # MSE loss to NaN in a single step.
        #
        # But exactly zero is the opposite failure, not a safe starting point.
        # The energy is quadratic in the charges, so dE/dq_k = 2 sum_j q_j B0_kj
        # vanishes identically at q = 0, and with the readout weight also zero
        # dE/dh = W^T dE/dq is zero as well, so nothing upstream is reached
        # either. Every gradient in charge_net is then exactly zero and stays
        # there: the term is not "starting at zero and growing from data", it is
        # dead for the whole run. The same argument kills the multipole terms,
        # E_qd being bilinear in (q, mu) and E_dd quadratic in mu.
        #
        # A small random weight breaks the symmetry so the gradient is nonzero,
        # while keeping the initial contribution negligible -- of order 1e-4 eV
        # against reference energies in the thousands.
        with torch.no_grad():
            nn.init.normal_(self.charge_net[-1].weight, std=init_readout_std)
            if self.charge_net[-1].bias is not None:
                self.charge_net[-1].bias.zero_()

        self.use_multipoles = use_multipoles
        if use_multipoles:
            # Learned invariant magnitude for the equivariant dipole, initialized
            # small-random for the same reason as the charge readout above.
            self.dipole_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            with torch.no_grad():
                nn.init.normal_(self.dipole_net[-1].weight, std=init_readout_std)
                if self.dipole_net[-1].bias is not None:
                    self.dipole_net[-1].bias.zero_()

        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("lr_cutoff", torch.tensor(lr_cutoff))
        self.ewald_dl = ewald_dl

    def forward(
        self,
        node_feats: torch.Tensor,
        orbital_positions: torch.Tensor,
        batch_orb: torch.Tensor,
        num_graphs: int,
        cell: Optional[torch.Tensor] = None,
        has_pbc: bool = False,
        dipole_dir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        charges = self.charge_net(node_feats).squeeze(-1)
        alpha = self.alpha

        use_mp = self.use_multipoles and dipole_dir is not None
        if use_mp:
            dip_scale = self.dipole_net(node_feats).squeeze(-1)
            # Equivariant dipole: invariant magnitude x frozen polarization
            # direction (detached so forces flow only through positions).
            dipoles = dip_scale.unsqueeze(-1) * dipole_dir.detach().to(node_feats.dtype)

        lr_energy = torch.zeros(num_graphs, device=node_feats.device,
                                dtype=node_feats.dtype)

        for g in range(num_graphs):
            mask = batch_orb == g
            q = charges[mask]
            pos = orbital_positions[mask]
            n = q.shape[0]
            if n < 2:
                continue

            # (n, n, 3) with diff[i, j] = r_j - r_i. Note the order: unsqueeze(0)
            # varies j along dim 1, so this is the NEGATIVE of r_ij = r_i - r_j.
            # The charge-dipole term below is antisymmetric in that choice, so
            # the convention has to be read off here rather than assumed.
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            # NaN-safe distance: the diagonal (i == j) and any coincident
            # orbital pairs have diff == 0. torch.linalg.norm has gradient
            # v / |v| = 0 / 0 = NaN there; the outer diag_mask would zero
            # the energy contribution, but autograd has already produced a
            # NaN local gradient that contaminates second derivatives
            # (forces) of any global parameter that scales positions (e.g.
            # orbital_spread_scale). An additive epsilon inside the sqrt
            # keeps the derivative bounded for all i, j.
            r = torch.sqrt((diff * diff).sum(dim=-1) + 1.0e-8)

            diag_mask = ~torch.eye(n, dtype=torch.bool, device=r.device)
            maskf = diag_mask.to(q.dtype)
            erfc_term = torch.erfc(alpha * r)
            B0 = erfc_term / r
            pair_e = q.unsqueeze(0) * q.unsqueeze(1) * B0

            if use_mp:
                mu = dipoles[mask]  # (n, 3)
                g_exp = (2.0 * alpha / math.sqrt(math.pi)) * torch.exp(-(alpha * r) ** 2)
                r2 = r * r
                B1 = (B0 + g_exp) / r2
                B2 = (3.0 * B1 + 2.0 * alpha * alpha * g_exp) / r2
                mi_d = torch.einsum("ik,ijk->ij", mu, diff)   # -mu_i . r_ij
                mj_d = torch.einsum("jk,ijk->ij", mu, diff)   # -mu_j . r_ij
                mimj = mu @ mu.t()                            # mu_i . mu_j
                # Expands to q_i (mu_j . r_ij) - q_j (mu_i . r_ij), the correct
                # multipole ordering, because both mi_d and mj_d carry the sign
                # flip from diff. e_dd is even in it and so is unaffected.
                e_qd = (q.unsqueeze(0) * mi_d - q.unsqueeze(1) * mj_d) * B1
                e_dd = mimj * B1 - mi_d * mj_d * B2
                pair_e = pair_e + e_qd + e_dd

            lr_energy[g] = 0.5 * (pair_e * maskf).sum()

            if has_pbc and cell is not None:
                cell_g = cell[g]
                volume = torch.abs(torch.linalg.det(cell_g))
                if volume > 1e-6:
                    # Reciprocal-space term is monopole-only; the dipole/
                    # quadrupole reciprocal contributions are omitted (the
                    # dataset is open-boundary molecules where the real-space
                    # multipole terms above are the relevant correction).
                    lr_energy[g] += self._ewald_reciprocal(q, pos, cell_g, volume, alpha)
                    lr_energy[g] -= alpha / math.sqrt(math.pi) * (q ** 2).sum()

        return lr_energy

    def _ewald_reciprocal(self, q, pos, cell, volume, alpha):
        inv_cell = torch.linalg.inv(cell)
        recip = 2.0 * math.pi * inv_cell.T

        n_max = max(1, int(self.ewald_dl))
        energy = torch.tensor(0.0, device=q.device, dtype=q.dtype)

        for nx in range(-n_max, n_max + 1):
            for ny in range(-n_max, n_max + 1):
                for nz in range(-n_max, n_max + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue
                    k = nx * recip[0] + ny * recip[1] + nz * recip[2]
                    k2 = torch.dot(k, k)
                    if k2 < 1e-10:
                        continue
                    kr = pos @ k
                    sf_real = (q * torch.cos(kr)).sum()
                    sf_imag = (q * torch.sin(kr)).sum()
                    sf2 = sf_real ** 2 + sf_imag ** 2
                    energy += sf2 * torch.exp(-k2 / (4.0 * alpha ** 2)) / k2

        energy *= 2.0 * math.pi / volume
        return energy
