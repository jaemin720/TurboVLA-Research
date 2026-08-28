from __future__ import annotations

import unittest

import torch

from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.decalign import DecAlignConfig, TriModalDecAlign


class TriModalDecAlignTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = DecAlignConfig(
            enabled=True,
            num_layers=1,
            num_heads=4,
            num_prototypes=3,
            gmm_em_iters=2,
            ot_num_iters=5,
            pde_dim=8,
            attention_dropout=0.0,
            relu_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            fusion_dropout=0.0,
        )
        self.model = TriModalDecAlign(hidden_dim=16, config=self.config)
        self.language = torch.randn(8, 5, 16)
        self.proprio = torch.randn(8, 2, 16)
        self.vision = torch.randn(8, 7, 16)
        self.language_mask = torch.tensor([[True, True, True, False, False]]).expand(8, -1).clone()

    def test_token_count_and_summary_shape(self) -> None:
        output = self.model(
            language_tokens=self.language,
            proprio_tokens=self.proprio,
            vision_tokens=self.vision,
            language_valid_mask=self.language_mask,
        )
        self.assertEqual(output.action_memory.shape, (8, 14, 16))
        self.assertEqual(output.summary_representation.shape, (8, 6 * 16))
        self.assertTrue(torch.equal(output.action_memory[:, 7 + 3 : 7 + 5], torch.zeros(8, 2, 16)))

    def test_objectives_are_finite_and_differentiable(self) -> None:
        language = self.language.clone().requires_grad_(True)
        proprio = self.proprio.clone().requires_grad_(True)
        vision = self.vision.clone().requires_grad_(True)
        output = self.model(
            language_tokens=language,
            proprio_tokens=proprio,
            vision_tokens=vision,
            language_valid_mask=self.language_mask,
            compute_objectives=True,
        )
        losses = output.losses
        total = (
            output.action_memory.square().mean()
            + losses.decoupling
            + 0.05 * losses.heterogeneous
            + 0.05 * losses.homogeneous
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(language.grad)
        self.assertIsNotNone(proprio.grad)
        self.assertIsNotNone(vision.grad)

    def test_padding_values_do_not_change_valid_outputs(self) -> None:
        self.model.eval()
        changed = self.language.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 1000.0
        with torch.no_grad():
            reference = self.model(
                self.language,
                self.proprio,
                self.vision,
                self.language_mask,
            ).action_memory
            actual = self.model(
                changed,
                self.proprio,
                self.vision,
                self.language_mask,
            ).action_memory
        torch.testing.assert_close(actual, reference)

    def test_config_round_trip_keeps_decalign_settings(self) -> None:
        config = TurboVLAConfig(decalign=self.config)
        restored = TurboVLAConfig.from_mapping(config.to_dict())
        self.assertTrue(restored.decalign.enabled)
        self.assertEqual(restored.decalign.num_layers, 1)
        self.assertEqual(restored.decalign.num_prototypes, 3)


if __name__ == "__main__":
    unittest.main()
