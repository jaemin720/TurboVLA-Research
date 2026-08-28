# Tri-modal DecAlign integration

This branch integrates the representation-learning core of
[TACO-Group/DecAlign](https://github.com/taco-group/DecAlign) into TurboVLA.
The imported ideas/code are MIT licensed; the preserved license text is in
`third_party/licenses/DecAlign-MIT.txt`.

## Scope

The integration uses three modalities:

| Original DecAlign slot | TurboVLA modality | Input shape for LIBERO |
| --- | --- | --- |
| language | BERT language tokens | `[B, 21, 256]` |
| audio | projected proprioception tokens | `[B, 2, 256]` |
| vision | DINOv3 patch tokens from two views | `[B, 512, 256]` |

TurboVLA's released bidirectional `VisionLanguageInteraction` is bypassed in
DecAlign mode.  DecAlign's three modality-specific heterogeneous Transformers
become the only multimodal interaction module.

## Data flow

```text
DINOv3 + VisionProjection ------------------- vision tokens (512)
BERT + TextProjection ----------------------- language tokens (21)
ActionHead.StateProjection ----------------- proprio tokens (2)
                                                     |
                                                     v
                                           TriModalDecAlign
                                                     |
                  unique/common token fusion, preserving each sequence length
                                                     |
                                                     v
                    [vision 512][language 21][proprio 2] = 535 memory tokens
                                                     |
                                                     v
                                            existing ACT decoder
                                                     |
                                                     v
                                             actions [B, 12, 7]
```

`StateProjection` remains under `action_head` so released parameter names stay
compatible.  In DecAlign mode it is called before DecAlign, and the resulting
proprio tokens are **not** appended again by the action head.

## What was preserved from DecAlign

- separate 1x1-equivalent linear encoders for modality-unique features;
- one weight-shared encoder for modality-common features;
- squared cosine decoupling objective;
- one heterogeneous Transformer per modality, with the other two unique
  sequences concatenated as context;
- per-batch diagonal-GMM fitting with EM;
- Gaussian prototype costs and three-way multi-marginal Sinkhorn transport;
- bidirectional local sample-to-cross-modal-prototype costs;
- modality-specific probabilistic distribution encoders (PDEs);
- mean, covariance, skew, and MMD common-distribution alignment;
- the six pooled summaries in language/proprio/vision unique-first order.

## Deliberate TurboVLA adaptations

1. **Audio becomes proprioception.**  The original audio slot is occupied by
   normalized robot state encoded into two tokens.
2. **Encoders remain external.**  DecAlign receives TurboVLA's online BERT,
   DINOv3, and state-projection outputs; its sentiment BERT/data pipeline is not
   imported.
3. **Attention is mask-aware.**  The original custom attention has no key
   padding mask.  The adaptation uses batch-first `nn.MultiheadAttention` while
   retaining the original post-norm residual/FFN order and cross-modal context.
4. **The action readout preserves tokens.**  Instead of pooling each modality
   into two vectors for prediction, transformed-unique and common features are
   fused channel-wise at every token.  The exact baseline memory length/order is
   preserved.  The paper-style six-summary vector is still returned for future
   summary-only ablations and diagnostics.
5. **Objectives are FP32 and training-only.**  GMM, Sinkhorn, covariance, skew,
   and MMD run outside BF16 autocast.  They are skipped during rollout because
   prototypes are training objectives rather than policy inputs.
6. **The task head changes.**  Sentiment regression/classification is replaced
   by TurboVLA's ACT decoder and masked action L1 loss.
7. **Supervised contrastive loss is omitted.**  LIBERO actions do not provide a
   direct counterpart to DecAlign's sentiment-class positives.

## Code map

| File | Responsibility |
| --- | --- |
| `turbovla/models/decalign/configuration.py` | DecAlign architecture and numerical settings |
| `turbovla/models/decalign/components.py` | masked pooling, cross-modal Transformer, token fusion |
| `turbovla/models/decalign/objectives.py` | decoupling, GMM/Sinkhorn, PDE/statistics/MMD |
| `turbovla/models/decalign/model.py` | tri-modal orchestration and 535-token readout |
| `turbovla/models/decalign/outputs.py` | typed representations and raw loss values |
| `turbovla/models/turbovla.py` | baseline/DecAlign routing |
| `turbovla/models/action_head.py` | separate state encoding and memory decoding APIs |
| `turbovla/training/trainer.py` | CLI, auxiliary loss composition, logging/checkpoint metadata |

## Training options

Enable the integration with:

```text
--use_decalign
--vla_feature_enhancer_layers 0
--no_require_feature_enhancer_preload
```

When this repository is a linked worktree while the `turbovla-libero`
environment is editable-installed from `/home/jack/TurboVLA`, run training from
the worktree root with the module entry point:

```text
cd /home/jack/TurboVLA-DecAlign
python -m turbovla.training.train_mixed ...
```

The current working directory then takes precedence over the editable install.
For simultaneous baseline and DecAlign jobs, separate environments/editable
installs are safer than relying on working-directory import precedence.

`build_model_architecture` forces the saved interaction layer count to zero in
DecAlign mode even if the explicit layer flag is omitted.  The
`--no_require_feature_enhancer_preload` flag is still required because there is
no bidirectional enhancer module into which released enhancer weights can be
loaded.

The initial paper-derived structural defaults are:

```text
DecAlign Transformer layers: 4
attention heads:             8
GMM prototypes:              7
GMM EM iterations:           5
Sinkhorn iterations:         50
OT regularization:           0.1
PDE dimension:               64
```

The total training objective is:

```text
action_l1
+ decalign_decoupling_weight    * decoupling
+ decalign_heterogeneous_weight * heterogeneous
+ decalign_homogeneous_weight   * homogeneous
```

Raw and weighted-independent component values are logged and stored in each
checkpoint.  `--decalign_loss_warmup_steps` can linearly warm up all three
auxiliary terms.

## Checkpoint behavior

- A checkpoint without a `decalign` config builds the unchanged baseline model
  and retains strict-load behavior.
- A newly trained DecAlign checkpoint serializes the complete nested DecAlign
  config and strict-loads the DecAlign architecture during evaluation.
- Released feature-enhancer parameters are intentionally not loaded in
  DecAlign mode.
- DecAlign training initializes every directly matching encoder, projection,
  `StateProjection`, and ACT decoder tensor from the supplied baseline
  checkpoint.  Only the removed interaction is discarded and new `decalign.*`
  tensors remain newly initialized.
- `StateProjection` and ACT decoder parameter names remain unchanged.

## Known experimental constraints

- GMM prototypes are fitted independently on each DDP rank, matching the
  original per-batch design rather than using global prototypes.
- Per-GPU batch size should preferably be at least `num_prototypes`; smaller
  batches are supported by repeating initialization samples but are less stable.
- Language instructions may repeat within a batch while vision/proprio vary,
  which can drive language variance to the configured floor.
- The 535-token readout is a TurboVLA-specific adaptation.  A future
  `summary_tokens` readout should use the already returned six pooled summaries
  and be evaluated as a separate checkpoint/ablation.

## Verification

`tests/test_decalign.py` covers shape preservation, finite objective/backward
computation, padding invariance, and config serialization without loading BERT
or DINOv3 weights.
