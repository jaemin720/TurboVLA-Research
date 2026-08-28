"""DecAlign training objectives adapted from TACO-Group/DecAlign (MIT)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .components import masked_mean
from .configuration import DecAlignConfig
from .outputs import DecAlignLosses


MODALITIES = ("language", "proprio", "vision")


class DistributionEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class DecAlignObjectives(nn.Module):
    """Paper-style decoupling, heterogeneous, and homogeneous objectives."""

    def __init__(self, hidden_dim: int, config: DecAlignConfig) -> None:
        super().__init__()
        self.config = config
        self.distribution_encoders = nn.ModuleDict(
            {
                name: DistributionEncoder(hidden_dim, config.pde_dim, config.fusion_dropout)
                for name in MODALITIES
            }
        )

    @staticmethod
    def _decoupling_loss(
        unique: torch.Tensor,
        common: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(dtype=unique.dtype)
        unique_flat = (unique * mask).flatten(1)
        common_flat = (common * mask).flatten(1)
        return F.cosine_similarity(unique_flat, common_flat, dim=1).pow(2).mean()

    def _init_gmm(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, dimension = samples.shape
        prototypes = self.config.num_prototypes
        if batch_size >= prototypes:
            indices = torch.linspace(0, batch_size - 1, steps=prototypes, device=samples.device).round().long()
            means = samples.index_select(0, indices)
        else:
            repeat = math.ceil(prototypes / batch_size)
            means = samples.repeat(repeat, 1)[:prototypes]
        variance = samples.var(dim=0, unbiased=False).clamp_min(self.config.gmm_var_floor)
        variance = variance.unsqueeze(0).expand(prototypes, dimension).clone()
        weights = torch.full(
            (prototypes,),
            1.0 / prototypes,
            device=samples.device,
            dtype=samples.dtype,
        )
        return weights, means, variance

    @staticmethod
    def _gmm_log_prob(
        samples: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        difference = samples.unsqueeze(1) - means.unsqueeze(0)
        mahalanobis = (difference.square() / variance.unsqueeze(0)).sum(dim=2)
        log_det = variance.log().sum(dim=1).unsqueeze(0)
        log_norm = samples.shape[1] * math.log(2.0 * math.pi)
        return weights.clamp_min(1e-8).log().unsqueeze(0) - 0.5 * (mahalanobis + log_det + log_norm)

    def _fit_gmm(self, samples: torch.Tensor) -> dict[str, torch.Tensor]:
        weights, means, variance = self._init_gmm(samples)
        for _ in range(self.config.gmm_em_iters):
            posterior = F.softmax(self._gmm_log_prob(samples, weights, means, variance), dim=1)
            counts = posterior.sum(dim=0).clamp_min(1e-6)
            weights = counts / samples.shape[0]
            means = posterior.transpose(0, 1).matmul(samples) / counts.unsqueeze(1)
            second_moment = posterior.transpose(0, 1).matmul(samples.square()) / counts.unsqueeze(1)
            variance = (second_moment - means.square()).clamp_min(self.config.gmm_var_floor)
        posterior = F.softmax(self._gmm_log_prob(samples, weights, means, variance), dim=1)
        return {"weights": weights, "means": means, "variance": variance, "posterior": posterior}

    @staticmethod
    def _gaussian_cost(
        means_a: torch.Tensor,
        variance_a: torch.Tensor,
        means_b: torch.Tensor,
        variance_b: torch.Tensor,
    ) -> torch.Tensor:
        mean_cost = (means_a.unsqueeze(1) - means_b.unsqueeze(0)).square().mean(dim=2)
        covariance_cost = (
            variance_a.unsqueeze(1)
            + variance_b.unsqueeze(0)
            - 2.0 * torch.sqrt(variance_a.unsqueeze(1) * variance_b.unsqueeze(0) + 1e-8)
        ).mean(dim=2)
        return mean_cost + covariance_cost

    @staticmethod
    def _sample_to_prototype_cost(samples: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        return (samples.unsqueeze(1) - means.unsqueeze(0)).square().mean(dim=2)

    def _multi_marginal_sinkhorn(
        self,
        cost: torch.Tensor,
        marginal_l: torch.Tensor,
        marginal_p: torch.Tensor,
        marginal_v: torch.Tensor,
    ) -> torch.Tensor:
        prototypes = cost.shape[0]
        regularization = self.config.ot_regularization
        kernel = torch.exp(-cost / regularization).clamp_min(1e-30)
        u = torch.ones(prototypes, device=cost.device, dtype=cost.dtype)
        v = torch.ones_like(u)
        w = torch.ones_like(u)
        for _ in range(self.config.ot_num_iters):
            u = marginal_l / (
                (kernel * v.view(1, prototypes, 1) * w.view(1, 1, prototypes)).sum(dim=(1, 2)) + 1e-8
            )
            v = marginal_p / (
                (kernel * u.view(prototypes, 1, 1) * w.view(1, 1, prototypes)).sum(dim=(0, 2)) + 1e-8
            )
            w = marginal_v / (
                (kernel * u.view(prototypes, 1, 1) * v.view(1, prototypes, 1)).sum(dim=(0, 1)) + 1e-8
            )
        plan = u.view(prototypes, 1, 1) * v.view(1, prototypes, 1) * w.view(1, 1, prototypes) * kernel
        entropy = (plan * (plan.clamp_min(1e-8).log() - 1.0)).sum()
        return (plan * cost).sum() + regularization * entropy

    def _heterogeneous_loss(
        self,
        unique: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = {name: masked_mean(unique[name], masks[name]) for name in MODALITIES}
        gmms = {name: self._fit_gmm(pooled[name]) for name in MODALITIES}
        marginals = {}
        for name in MODALITIES:
            marginal = gmms[name]["posterior"].mean(dim=0)
            marginals[name] = marginal / marginal.sum().clamp_min(1e-8)

        cost_lp = self._gaussian_cost(
            gmms["language"]["means"],
            gmms["language"]["variance"],
            gmms["proprio"]["means"],
            gmms["proprio"]["variance"],
        )
        cost_lv = self._gaussian_cost(
            gmms["language"]["means"],
            gmms["language"]["variance"],
            gmms["vision"]["means"],
            gmms["vision"]["variance"],
        )
        cost_pv = self._gaussian_cost(
            gmms["proprio"]["means"],
            gmms["proprio"]["variance"],
            gmms["vision"]["means"],
            gmms["vision"]["variance"],
        )
        joint_cost = cost_lp.unsqueeze(2) + cost_lv.unsqueeze(1) + cost_pv.unsqueeze(0)
        prototype_ot = self._multi_marginal_sinkhorn(
            joint_cost,
            marginals["language"],
            marginals["proprio"],
            marginals["vision"],
        )

        local_terms = []
        for source in MODALITIES:
            for target in MODALITIES:
                if source == target:
                    continue
                costs = self._sample_to_prototype_cost(pooled[source], gmms[target]["means"])
                local_terms.append((gmms[source]["posterior"] * costs).sum(dim=1).mean())
        local_prototype = torch.stack(local_terms).mean()
        return prototype_ot + local_prototype, prototype_ot, local_prototype

    @staticmethod
    def _distribution_stats(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = latent.mean(dim=0)
        centered = latent - mean
        denominator = max(latent.shape[0] - 1, 1)
        covariance = centered.transpose(0, 1).matmul(centered) / denominator
        variance = centered.square().mean(dim=0).clamp_min(1e-6)
        skew = centered.pow(3).mean(dim=0) / variance.pow(1.5)
        return mean, covariance, skew

    @staticmethod
    def _stats_distance(
        stats_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stats_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return sum(F.mse_loss(item_a, item_b) for item_a, item_b in zip(stats_a, stats_b))

    def _mmd(self, samples_a: torch.Tensor, samples_b: torch.Tensor) -> torch.Tensor:
        norm_a = samples_a.square().sum(dim=1, keepdim=True)
        norm_b = samples_b.square().sum(dim=1, keepdim=True)
        distance_aa = norm_a + norm_a.transpose(0, 1) - 2.0 * samples_a.matmul(samples_a.transpose(0, 1))
        distance_bb = norm_b + norm_b.transpose(0, 1) - 2.0 * samples_b.matmul(samples_b.transpose(0, 1))
        distance_ab = norm_a + norm_b.transpose(0, 1) - 2.0 * samples_a.matmul(samples_b.transpose(0, 1))
        gamma = 1.0 / (2.0 * self.config.mmd_bandwidth)
        return (
            torch.exp(-gamma * distance_aa.clamp_min(0.0)).mean()
            + torch.exp(-gamma * distance_bb.clamp_min(0.0)).mean()
            - 2.0 * torch.exp(-gamma * distance_ab.clamp_min(0.0)).mean()
        )

    def _homogeneous_loss(
        self,
        common: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = {
            name: self.distribution_encoders[name](masked_mean(common[name], masks[name]))
            for name in MODALITIES
        }
        stats = {name: self._distribution_stats(latent[name]) for name in MODALITIES}
        pairs = (("language", "proprio"), ("language", "vision"), ("proprio", "vision"))
        statistics = torch.stack([self._stats_distance(stats[a], stats[b]) for a, b in pairs]).mean()
        mmd = torch.stack([self._mmd(latent[a], latent[b]) for a, b in pairs]).mean()
        return statistics + mmd, statistics, mmd

    def forward(
        self,
        unique: dict[str, torch.Tensor],
        common: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> DecAlignLosses:
        # TurboVLA adaptation: GMM, Sinkhorn, covariance, skew, and MMD stay in
        # FP32 even when encoder/action computation is autocast to BF16.
        device_type = unique["language"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            unique_fp32 = {name: value.float() for name, value in unique.items()}
            common_fp32 = {name: value.float() for name, value in common.items()}
            decoupling = torch.stack(
                [self._decoupling_loss(unique_fp32[name], common_fp32[name], masks[name]) for name in MODALITIES]
            ).mean()
            heterogeneous, prototype_ot, local_prototype = self._heterogeneous_loss(unique_fp32, masks)
            homogeneous, distribution_statistics, mmd = self._homogeneous_loss(common_fp32, masks)
        return DecAlignLosses(
            decoupling=decoupling,
            heterogeneous=heterogeneous,
            homogeneous=homogeneous,
            prototype_ot=prototype_ot,
            local_prototype=local_prototype,
            distribution_statistics=distribution_statistics,
            mmd=mmd,
        )
