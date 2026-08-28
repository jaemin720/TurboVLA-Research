# TurboVLA Research

[![Upstream](https://img.shields.io/badge/Upstream-TurboVLA-181717?logo=github)](https://github.com/H-EmbodVis/TurboVLA)
[![License](https://img.shields.io/badge/License-Apache%202.0-2563eb)](LICENSE)

> [!IMPORTANT]
> This is an **unofficial, independently maintained research repository** built on
> [TurboVLA](https://github.com/H-EmbodVis/TurboVLA). It is not affiliated with or maintained by the original authors.

This repository is a research workspace for developing and evaluating new vision-language-action (VLA) methods on top
of TurboVLA. The current baseline supports the original LIBERO benchmark and the five standard LIBERO-PRO perturbation
suites for LIBERO-10.

No project-specific performance claims are reported yet. Results published by the original TurboVLA project are upstream
results, not results produced by this repository.

## Project status

- [x] TurboVLA checkpoint and backbone loading
- [x] Standard LIBERO rollout evaluation
- [x] LIBERO-PRO object, position, semantic, task, and environment suites
- [x] Independent per-command selection of LIBERO and LIBERO-PRO
- [x] Tri-modal token-preserving DecAlign integration (experimental)
- [ ] Training and ablation experiments
- [ ] Reproduced benchmark table

The experimental DecAlign architecture, its differences from the source
implementation, training flags, checkpoint behavior, and verification status
are documented in [DECALIGN_INTEGRATION.md](DECALIGN_INTEGRATION.md).

## What this repository adds

The evaluation adapter registers the following LIBERO-PRO variants for the LIBERO-10 `long.pth` checkpoint:

| Generalization dimension | Suite name |
| --- | --- |
| Object | `libero_10_object` |
| Position | `libero_10_swap` |
| Semantic / language | `libero_10_lan` |
| Task | `libero_10_task` |
| Environment | `libero_10_env` |

The evaluator selects benchmark code through `--libero_root` and benchmark assets through `LIBERO_CONFIG_PATH`. This
keeps the original LIBERO and LIBERO-PRO source trees isolated even though both expose a Python package named `libero`.

## Reproduce the current setup

The instructions below capture the setup that was validated for this repository, including the dependency and renderer
issues encountered during reproduction.

### 1. Recommended directory layout

Keep the benchmark repositories and datasets outside this repository:

```text
/path/to/workspace/
├── TurboVLA-Research/
├── LIBERO/
└── LIBERO-PRO/

/path/to/libero_datasets/
└── libero_10/
```

Set reusable paths after cloning:

```bash
git clone https://github.com/jaemin720/TurboVLA-Research.git
cd TurboVLA-Research

export TURBOVLA_ROOT="$PWD"
export LIBERO_ROOT=/absolute/path/to/LIBERO
export LIBERO_PRO_ROOT=/absolute/path/to/LIBERO-PRO
```

### 2. Create the TurboVLA environment

```bash
conda create -n turbovla-libero python=3.10 -y
conda activate turbovla-libero

python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[libero]"

# Required by the validated LIBERO simulator stack.
python -m pip install matplotlib==3.5.3 mujoco==2.3.7
```

Validated core versions:

| Package | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| transformers | 4.57.6 |
| timm | 1.0.28 |
| robosuite | 1.4.1 |
| MuJoCo | 2.3.7 |
| matplotlib | 3.5.3 |

The exact TensorFlow and utility-package patch versions are not part of the simulator compatibility contract. Use the
versions resolved by `pip install -e ".[libero]"` unless an experiment requires a stricter lock file.

### 3. Download checkpoints and backbones

Authenticate with Hugging Face and download the released TurboVLA checkpoint:

```bash
hf auth login

hf download H-EmbodVis/TurboVLA \
  --local-dir pretrained/TurboVLA
```

Download DINOv3 ViT-B into an explicit local directory. The DINOv3 repository may require accepting its access terms:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir pretrained/backbones/dinov3-vitb16
```

For reproducible offline evaluation, also materialize BERT outside the Hugging Face cache:

```bash
hf download google-bert/bert-base-uncased \
  --local-dir pretrained/backbones/bert-base-uncased
```

Set the backbone paths:

```bash
export DINOV3_ROOT="$TURBOVLA_ROOT/pretrained/backbones/dinov3-vitb16"
export BERT_ROOT="$TURBOVLA_ROOT/pretrained/backbones/bert-base-uncased"
```

Expected files include:

```text
pretrained/
├── TurboVLA/checkpoints/libero/long.pth
└── backbones/
    ├── dinov3-vitb16/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    └── bert-base-uncased/
```

Model weights and datasets are ignored by Git and must not be committed to this repository.

### 4. Prepare standard LIBERO

Clone the original benchmark:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
```

Install only the LIBERO source package. Do not let it replace the TurboVLA dependency set:

```bash
python -m pip install -e "$LIBERO_ROOT" \
  --no-deps \
  --no-build-isolation \
  --config-settings editable_mode=compat
```

The compatibility editable mode is important for LIBERO's nested package layout. A normal editable install can report
success while `import libero` still fails.

Create a benchmark-specific configuration directory:

```bash
mkdir -p "$HOME/.config/libero/turbovla"
nano "$HOME/.config/libero/turbovla/config.yaml"
```

Use absolute paths in `config.yaml`:

```yaml
benchmark_root: /absolute/path/to/LIBERO/libero/libero
assets: /absolute/path/to/LIBERO/libero/libero/assets
bddl_files: /absolute/path/to/LIBERO/libero/libero/bddl_files
init_states: /absolute/path/to/LIBERO/libero/libero/init_files
datasets: /absolute/path/to/libero_datasets
```

Verify the import and configuration before loading the policy:

```bash
LIBERO_CONFIG_PATH="$HOME/.config/libero/turbovla" \
python -c "from libero.libero import benchmark, get_libero_path; \
print(get_libero_path('bddl_files')); \
print(get_libero_path('init_states')); \
print(benchmark.get_benchmark_dict()['libero_10']().n_tasks)"
```

The final line should print `10`. An interactive dataset-path prompt means that
`$LIBERO_CONFIG_PATH/config.yaml` is missing.

### 5. Validate the TurboVLA model

Strict-load the checkpoint before starting the simulator:

```bash
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/long.pth \
  --libero_root "$LIBERO_ROOT" \
  --dinov3_path "$DINOV3_ROOT" \
  --bert_path "$BERT_ROOT" \
  --allow_hf_download false \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_10 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --dry_run_model_load true
```

A successful dry run reports a strict checkpoint load and exits before creating a LIBERO environment.

### 6. Run a standard LIBERO smoke test

Use EGL on NVIDIA GPUs. The default OSMesa path can produce LLVM `Required pass not found` errors on some systems.

```bash
LIBERO_CONFIG_PATH="$HOME/.config/libero/turbovla" \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
USE_TF=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/long.pth \
  --libero_root "$LIBERO_ROOT" \
  --dinov3_path "$DINOV3_ROOT" \
  --bert_path "$BERT_ROOT" \
  --allow_hf_download false \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_10 \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --mujoco_gl egl \
  --pyopengl_platform egl \
  --result_json_path outputs/evaluation/libero_10_smoke.json
```

The smoke test validates model loading, environment creation, rendering, action rollout, and result serialization. An
episode with `success=False` is a valid policy outcome and is not an execution failure.

### 7. Prepare LIBERO-PRO

Clone LIBERO-PRO separately:

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$LIBERO_PRO_ROOT"
```

Do **not** run either of the following commands in `turbovla-libero`:

```bash
# Do not run these in the TurboVLA environment.
pip install -r "$LIBERO_PRO_ROOT/requirements.txt"
pip install -e "$LIBERO_PRO_ROOT"
```

LIBERO-PRO pins an older standalone environment, and an editable install would also compete with the original `libero`
package. The evaluator loads LIBERO-PRO directly through `--libero_root`, so no package installation is needed.

Download the official pre-generated BDDL and initialization files:

```bash
hf download zhouxueyang/LIBERO-Pro \
  --repo-type dataset \
  --include "bddl_files/**" "init_files/**" \
  --local-dir "$LIBERO_PRO_ROOT/libero_data"

cp -a "$LIBERO_PRO_ROOT/libero_data/bddl_files/." \
  "$LIBERO_PRO_ROOT/libero/libero/bddl_files/"
cp -a "$LIBERO_PRO_ROOT/libero_data/init_files/." \
  "$LIBERO_PRO_ROOT/libero/libero/init_files/"
```

Create a separate LIBERO-PRO configuration:

```bash
mkdir -p "$HOME/.config/libero/turbovla-pro"
nano "$HOME/.config/libero/turbovla-pro/config.yaml"
```

```yaml
benchmark_root: /absolute/path/to/LIBERO-PRO/libero/libero
assets: /absolute/path/to/LIBERO-PRO/libero/libero/assets
bddl_files: /absolute/path/to/LIBERO-PRO/libero/libero/bddl_files
init_states: /absolute/path/to/LIBERO-PRO/libero/libero/init_files
datasets: /absolute/path/to/libero_datasets
```

The official pre-generated files are sufficient for the five standard perturbation suites. LIBERO-PRO's dynamic
`perturbation.py` generator is only needed for custom perturbation combinations.

### 8. Run a LIBERO-PRO smoke test

```bash
LIBERO_CONFIG_PATH="$HOME/.config/libero/turbovla-pro" \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
USE_TF=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/long.pth \
  --libero_root "$LIBERO_PRO_ROOT" \
  --dinov3_path "$DINOV3_ROOT" \
  --bert_path "$BERT_ROOT" \
  --allow_hf_download false \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_10_object \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --mujoco_gl egl \
  --pyopengl_platform egl \
  --result_json_path outputs/evaluation/libero_pro_10_object_smoke.json
```

## Selecting a benchmark

Always change `LIBERO_CONFIG_PATH`, `--libero_root`, and `--task_suite_name` together:

| Evaluation | `LIBERO_CONFIG_PATH` | `--libero_root` | `--task_suite_name` |
| --- | --- | --- | --- |
| Standard LIBERO-10 | `.../turbovla` | original LIBERO | `libero_10` |
| PRO Object | `.../turbovla-pro` | LIBERO-PRO | `libero_10_object` |
| PRO Position | `.../turbovla-pro` | LIBERO-PRO | `libero_10_swap` |
| PRO Semantic | `.../turbovla-pro` | LIBERO-PRO | `libero_10_lan` |
| PRO Task | `.../turbovla-pro` | LIBERO-PRO | `libero_10_task` |
| PRO Environment | `.../turbovla-pro` | LIBERO-PRO | `libero_10_env` |

For a full suite evaluation, remove `--task_ids 0` and use:

```text
--num_trials_per_task 50
```

Standard LIBERO-10 contains 10 tasks and 50 trials per task. Evaluating the five LIBERO-PRO dimensions therefore runs
2,500 episodes; adding the original LIBERO-10 baseline gives 3,000 episodes.

## Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'libero'` after editable install | LIBERO nested package layout | Reinstall with `editable_mode=compat` |
| Interactive dataset-path prompt | Missing config in the selected directory | Create `$LIBERO_CONFIG_PATH/config.yaml` |
| `No module named 'matplotlib'` | Missing LIBERO runtime dependency | Install `matplotlib==3.5.3` |
| Assertion failure around MuJoCo joint addresses | Incompatible modern MuJoCo version | Use `mujoco==2.3.7` with robosuite 1.4.1 |
| Repeated LLVM `Required pass not found` messages | OSMesa/LLVM renderer incompatibility | Use EGL with `--mujoco_gl egl --pyopengl_platform egl` |
| Missing `preprocessor_config.json` | A Hugging Face model ID was passed where a local DINO path was required | Download DINOv3 locally and pass that directory |
| Hugging Face retry or DNS errors during evaluation | Online lookup attempted despite cached assets | Use explicit local model paths, `--allow_hf_download false`, and offline variables |
| robosuite private macro warning | Optional local robosuite configuration is absent | Safe to ignore for evaluation |
| Gym maintenance warning | Legacy dependency used by LIBERO | Warning only; do not replace it during reproduction |

## Research workflow

Keep `main` runnable and develop unverified methods on a research branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c research/method-name
```

Commit source code and lightweight experiment configuration. Do not commit pretrained models, datasets, videos, logs,
tokens, or machine-specific absolute paths.

The original TurboVLA repository is retained as the `upstream` remote:

```bash
git fetch upstream
git merge upstream/main
```

Review upstream changes before merging because they may overlap with research modifications.

## Upstream projects and attribution

This project builds on:

- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA) — base model, training code, checkpoints, and evaluation runtime
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — manipulation benchmark
- [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) — robustness and generalization benchmark
- [DINOv3](https://github.com/facebookresearch/dinov3) — visual encoder
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — vision-language interaction components
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) — LIBERO rollout protocol

Use the official repositories for authoritative pretrained artifacts and reported results.

## Citation

If this code is useful in your research, cite the original TurboVLA project and the benchmarks used:

```bibtex
@article{xie2026turbovla,
  title   = {TurboVLA: Real-Time Vision-Language-Action Model at
             32 Hz on an RTX 4090 with <1 GB VRAM},
  author  = {Xie, Hengyi and Yao, Chenfei and Wu, Xianjin and
             Xi, Xuanyang and Tang, Yiping and Xu, Di and
             Zhu, Yingying and Liang, Dingkang and Bai, Xiang and Ding, Han},
  journal = {arXiv preprint arXiv:2607.27205},
  year    = {2026}
}

@article{liu2023libero,
  title   = {LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author  = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and
             Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2023}
}

@article{zhou2025liberopro,
  title   = {LIBERO-PRO: Towards Robust and Fair Evaluation of
             Vision-Language-Action Models Beyond Memorization},
  author  = {Zhou, Xueyang and Xu, Yangming and Tie, Guiyao and
             Chen, Yongchao and Zhang, Guowen and Chu, Duanfeng and
             Zhou, Pan and Sun, Lichao},
  journal = {arXiv preprint arXiv:2510.03827},
  year    = {2025}
}
```

## License

This derivative repository retains the upstream [Apache License 2.0](LICENSE). Third-party components remain subject to
their respective licenses and attribution requirements.
