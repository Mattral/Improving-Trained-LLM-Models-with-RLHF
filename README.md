# RLHF Platform — PPO + DPO for LLM Alignment

> A modular RLHF training framework implementing PPO and DPO from scratch. Validated end-to-end on a single T4 GPU in under 20 minutes. Designed with a distributed execution architecture that scales to multi-node clusters.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](#requirements)
[![Tests 110+](https://img.shields.io/badge/Tests-110%2B-brightgreen.svg)](tests/)
[![Type Coverage 100%](https://img.shields.io/badge/Type%20Coverage-100%25-brightgreen.svg)](#code-quality)
[![CI](https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO/actions/workflows/ci_perf.yml/badge.svg)](https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO/actions)

---

## What this is

This repo implements the full RLHF alignment pipeline — SFT → Reward Model → PPO and/or DPO — with production-engineering discipline:

- **PPO engine** with Generalized Advantage Estimation, clipped surrogate objective, and adaptive KL penalty
- **DPO engine** implementing reference-free preference optimization without a reward model
- **Pydantic v2 config system** with strict validation, YAML/JSON serialization, and factory methods
- **Typer CLI** with four commands covering every training stage
- **Async dataset pipeline** with JSONL caching and toy mode (1K HH-RLHF samples, <20 min on T4)
- **110+ tests** (40+ PPO, 35+ DPO, 35+ config), 100% type-annotated, fully verified

The distributed architecture is designed for multi-node scale: asymmetric process groups separating the active training mesh (Actor + Critic, ZeRO-3) from the frozen inference ring (Reference + Reward), with async checkpointing and NCCL bucket overlapping.

---

## What's validated vs. what's designed

| Component | Status | Verified on |
|---|---|---|
| Pydantic v2 config system | ✅ Complete + tested | CPU (all hardware) |
| PPO engine (GAE + clip + KL) | ✅ Complete + 40+ tests | CPU unit tests |
| DPO engine | ✅ Complete + 35+ tests | CPU unit tests |
| CLI (4 commands) | ✅ Complete + tested | Single T4 GPU |
| SFT trainer (LoRA + HF Trainer) | ✅ Complete | Single T4 GPU (<20 min toy) |
| Reward model trainer | ✅ Complete | Single T4 GPU |
| Async checkpointing (`async_io.py`) | ✅ Complete | CPU tests |
| NCCL comm hooks (`comm_hooks.py`) | ✅ Implemented | Not multi-node validated |
| DeepSpeed ZeRO-3 topology | ✅ Config + code | Not multi-node validated |
| TRL benchmark comparison | ⚠️ Framework ready | Requires `pip install trl` |
| Multi-node cluster (128+ GPUs) | ❌ Not validated | Design only |

The distributed multi-node path — `torchrun --nnodes=128` — is architecturally sound and the code is written to support it, but has not been run on a cluster. The docs describe the intended behavior, not measured results.

---

## Quick start (single T4, under 20 minutes)

```bash
git clone https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO
cd Improving-LLM-Models-with-RLHF-PPO-DPO
pip install -e .

# 1. Fine-tune base model (5-7 min)
python -m rlhf_platform.cli train-sft --toy --epochs 1

# 2. Train reward model (3-5 min)
python -m rlhf_platform.cli train-reward --toy --epochs 1

# 3. PPO alignment
python -m rlhf_platform.cli run-ppo --toy

# 4. DPO alignment (alternative to PPO)
python -m rlhf_platform.cli run-dpo --toy
```

Toy mode uses 1,000 preference pairs from Anthropic HH-RLHF. No cluster required.

---

## Installation

Three requirements files for different components:

```bash
pip install -e .                              # Core PPO/DPO + CLI
pip install -r requirements-fine-tune.txt    # SFT trainer
pip install -r requirements-reward.txt       # Reward model trainer
pip install trl==0.5.0                       # Optional: TRL benchmark comparison
```

Run tests:
```bash
pytest tests/ -v                             # 110+ tests, CPU-only safe
CUDA_VISIBLE_DEVICES="" pytest tests/        # Force CPU if CUDA errors
```

---

## Architecture

The framework separates four models into two process groups to avoid the "generation bubble" — the idle GPU time caused by waiting for rollout generation before training can start.

```
ACTOR_CRITIC_GROUP (active optimization)
  ├── Actor (policy)    — DeepSpeed ZeRO-3, generates responses
  └── Critic (value)    — estimates returns for GAE

REFERENCE_REWARD_GROUP (frozen inference)
  ├── Reference model   — computes KL baseline (no gradient tracking)
  └── Reward model      — scores responses (no gradient tracking)

Host CPU pinned ring-buffer
  — async streaming between groups
  — pre-populates epoch N+1 rollouts while epoch N trains
```

While Actor/Critic run backprop for step N, the Reference/Reward group populates the next batch of rollouts. NCCL gradient buckets are reduced asynchronously during the backward pass rather than after it.

**Important:** This architecture is implemented and the design is sound. Multi-node validation (128+ GPUs) has not been performed.

---

## PPO implementation

The core loss function:

$$\mathcal{L}_{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right] - \beta\, D_{KL}\left(\pi_\theta \parallel \pi_{ref}\right)$$

Implemented in `src/rlhf_platform/alignment/`:

- `loss.py` — numerically stable KL penalties, clipped advantage computation
- `ppo_engine.py` — multi-model PPO step orchestration, W&B logging
- `rollout.py` — async generation pipeline, pinned memory buffer

GAE advantage estimation runs with distributed variance normalization across the full `ACTOR_CRITIC_GROUP` rank mesh.

---

## DPO implementation

DPO skips the reward model entirely. Given a preferred response $y_w$ and a rejected response $y_l$ for prompt $x$:

$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

The DPO engine (`src/rlhf_platform/alignment/dpo_engine.py`) is 382 lines, 100% type-safe, with 35+ unit tests. Run via `python -m rlhf_platform.cli run-dpo`.

---

## Core Systems Optimization Pillars

### 1. Asynchronous Rollout Ring-Buffers (`rollout.py`)

Auto-regressive token sampling is bound by memory bandwidth, while gradient updates are bound by matrix multiplication compute limits. Instead of executing these phases sequentially, our rollout engine utilizes an asynchronous background generator. While the active compute mesh executes backpropagation updates for epoch $N$, the inference mesh continuously populates a thread-safe, pinned CPU host memory ring buffer with rollout tokens for epoch $N+1$. This architecture entirely mitigates generation stalls.

### 2. NCCL Collective Communication Overlapping (`comm_hooks.py`)

During the Actor's backward pass, gradients are not cached globally until the end of the execution step. Instead, we register custom communication hooks. As independent layers finalize their gradients, they are immediately packed into discrete memory buckets. The engine triggers asynchronous network operations (`all_reduce` or `reduce_scatter`) over InfiniBand channels concurrently while the remaining GPU clusters continue executing preceding tensor layers.

### 3. Non-Blocking Fault Tolerance (`async_io.py`)

At petascale, Mean Time Between Failures (MTBF) degrades to hours. Traditional saving operations freeze the execution graph across all ranks, wasting millions of compute cycles. This engine leverages multi-tiered, asynchronous checkpointing: model weights are copied instantly to CPU pinned memory via local memory copies, and a background thread streams the snapshot to storage asynchronously while rank 0 handles disk IO, letting the primary cluster resume training within milliseconds.

---

## Configuration

All training behavior is controlled by `configs/cluster_topology.yaml` and `configs/deepspeed_zero3.yaml`. The Pydantic v2 config system validates every field at load time with clear error messages.

```python
from rlhf_platform.config import RLHFConfig

# Toy mode — single GPU, small model, 1K samples
config = RLHFConfig.toy_mode()

# Production mode — reads from YAML
config = RLHFConfig.from_yaml("configs/cluster_topology.yaml")
```

Topology must satisfy: `tensor_parallel × pipeline_parallel × data_parallel = WORLD_SIZE`. The config validator enforces this at startup.

---

## Distributed training (multi-node design)

```bash
torchrun \
  --nnodes=128 \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  train.py --config configs/cluster_topology.yaml
```

**Note:** This launch command is the intended interface. It has not been validated on a physical cluster. For single-GPU use, the CLI commands in Quick Start are the validated path.

Cluster configuration targets:

| Metric / Layer | 8x GPU Node (Local Dev) | 512x GPU Cluster (Pod Scale) | 10,000x GPU Cluster (Petascale) |
| --- | --- | --- | --- |
| **Tensor Parallelism (TP)** | 1 | 8 (Intra-Chassis NVLink) | 8 (Intra-Chassis NVLink) |
| **Pipeline Parallelism (PP)** | 1 | 2 (Inter-Node InfiniBand) | 16 (Inter-Node Ring) |
| **Data Parallelism (DP)** | 8 (ZeRO-3) | 32 (FSDP + Sharding) | 780 (Hybrid FSDP / ZeRO) |
| **Gradient Overlap Bucket** | 25MB | 50MB | 128MB |
| **Target Context Length** | 4,096 | 16,384 | 65,536 |

---

## Telemetry

Every training step emits structured JSON to stdout — connects directly to Grafana, Prometheus, or W&B:

```json
{
  "timestamp": "2026-05-29T21:44:45Z",
  "rank": 0,
  "step": 1420,
  "type": "ppo_step",
  "policy_loss": 0.0412,
  "value_loss": 0.1182,
  "kl_divergence": 0.0314,
  "vram_allocated_bytes": 79456891200,
  "nccl_bubble_stall_ms": 0.42,
  "tokens_per_sec_per_gpu": 2450.8
}
```

---

## Repository layout

```
src/rlhf_platform/
├── distributed/
│   ├── topology.py       # Process group creation, rank assignment
│   ├── comm_hooks.py     # NCCL bucket overlap hooks
│   └── async_io.py       # Non-blocking background checkpointing
├── alignment/
│   ├── loss.py           # KL penalty, clipped advantage (numerically stable)
│   ├── ppo_engine.py     # PPO step orchestration
│   ├── dpo_engine.py     # DPO loss + training loop
│   └── rollout.py        # Async generation + pinned ring buffer
└── utils/
    └── telemetry.py      # Rank-aware zero-allocation JSON metrics

configs/
├── cluster_topology.yaml # Training + parallelism hyperparameters
└── deepspeed_zero3.yaml  # ZeRO-3 optimizer config

docs/
├── core/ARCHITECTURE.md      # Component lifecycles, execution topology
├── core/philosophy.md        # Design tradeoffs
├── operations/system_design.md
├── operations/setup.md       # Cluster deployment runbook
├── governance/security.md    # Reward hacking, checkpoint security
└── governance/contributing.md

tests/                    # 110+ tests — PPO, DPO, config, integration
results/benchmarks.md     # Benchmark framework (TRL comparison ready)
```

---

## Documentation map

| Audience | Start here | Deep dive |
|---|---|---|
| First-time user | [Quick Start](#quick-start-single-t4-under-20-minutes) | [docs/PHASE_3_CLI.md](docs/PHASE_3_CLI.md) |
| ML engineer | [PPO implementation](#ppo-implementation) | [docs/core/ARCHITECTURE.md](docs/core/ARCHITECTURE.md) |
| Infra / DevOps | [Distributed training](#distributed-training-multi-node-design) | [docs/operations/setup.md](docs/operations/setup.md) |
| Contributor | [Contributing](docs/governance/contributing.md) | [DEVELOPMENT.md](docs/DEVELOPMENT.md) |

Phase-by-phase implementation docs: [PHASE_1_CONFIG.md](docs/PHASE_1_CONFIG.md) · [PHASE_2_PPO.md](docs/PHASE_2_PPO.md) · [PHASE_3_CLI.md](docs/PHASE_3_CLI.md) · [PHASE_4_BENCHMARKS.md](docs/PHASE_4_BENCHMARKS.md)

---

## Contributing

```bash
pre-commit install
pytest tests/ -v
mypy src/rlhf_platform --strict
black src/ tests/ && ruff check --fix
```

All PRs must pass: tests · mypy strict · black + ruff · no fabricated claims.

---

## Citation

```bibtex
@software{rlhf_platform_2026,
  title  = {RLHF Platform: PPO and DPO for LLM Alignment},
  author = {Mattral and contributors},
  year   = {2026},
  url    = {https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO}
}

@article{schulman2017proximal,
  title  = {Proximal Policy Optimization Algorithms},
  author = {Schulman, John and others},
  year   = {2017},
  url    = {https://arxiv.org/abs/1707.06347}
}

@article{rafailov2023direct,
  title  = {Direct Preference Optimization},
  author = {Rafailov, Rafael and others},
  year   = {2023},
  url    = {https://arxiv.org/abs/2305.18290}
}
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.


## Acknowledgments

This project builds on the work of:
- [OpenAI](https://openai.com/) for PPO research and baselines
- [Hugging Face](https://huggingface.co/) for TRL library and transformers
- [DeepSpeed](https://www.deepspeed.ai/) for distributed training optimization
- [Anthropic](https://www.anthropic.com/) for RLHF research and HH-RLHF dataset
- The open-source community for PyTorch, Pydantic, and ecosystem tools
