# Roadmap — RLHF Platform Productionization

Reference: original educational walkthrough moved to `docs/educational_deepdive.md`.

## Completed ✅

### Core Infrastructure
- Moved original educational README to `docs/educational_deepdive.md` (preserves original context).
- Implemented `src/rlhf_platform/distributed/topology.py` with:
  - Asymmetric multi-model placement grid (Actor/Critic active vs. Reference/Reward frozen)
  - Explicit `dist.new_group()` process group creation and barrier synchronization
  - Strategy parsing from YAML with fallback validation
  - Device mapping and rank validation
- Implemented `src/rlhf_platform/distributed/async_io.py` for non-blocking async checkpoint writes with pinned CPU memory staging.
- Implemented `src/rlhf_platform/distributed/comm_hooks.py` with gradient bucket overlapping, KL divergence checks, and NaN detection.

### Alignment & Training
- Implemented `src/rlhf_platform/alignment/rollout.py` with:
  - `RolloutBuffer` ring buffer using pinned CPU memory and thread-safety locks
  - `RolloutGenerator` autoregressive generation with temperature + top-p sampling
  - `RolloutCollator` batch padding and sequence alignment
  - `AsyncRolloutPipeline` background generator threads with non-blocking device transfers
- Implemented `src/rlhf_platform/alignment/ppo_engine.py` with:
  - Full DistributedPPOEngine orchestrating Actor, Critic, Reference, Reward models
  - Communication hook registration for active ranks
  - Numerical safety checks on aggregated metrics
  - Non-blocking device transfers for rollout batches
- Implemented `src/rlhf_platform/alignment/loss.py` with stable KL divergence, GAE advantages, PPO clipping, and value function regularization.

### Production Entrypoints & Validation
- **Enhanced `train.py`** with comprehensive initialization sequencing:
  - 5-stage initialization: distributed → topology → models → DDP wrapping → pipeline/engine
  - Rich logging with stage-based progress reporting and error diagnostics
  - Random seed management for reproducibility
  - Verbose mode for debugging distributed issues
  - Graceful error handling and trace dumps
  - Non-blocking device transfers and memory profiling
- **Created `scripts/simulate_runtime.py`** for multi-rank cluster emulation:
  - `torch.multiprocessing.spawn` launches 4-rank cluster on CPU with gloo backend
  - Validates asymmetric topology group creation without deadlocks
  - Tests thread-safe rollout buffer with pinned memory operations
  - Validates non-blocking tensor transfers across ranks
  - Generates rank-aware JSON trace logs for forensics
  - Exit code `0` on success, `1` on failure with full trace dumps
- **Created `scripts/README.md`** with usage guide and troubleshooting.

### Testing & Configuration
- Added unit tests for `topology` and `rollout` modules (`tests/`).
- Configured `configs/cluster_topology.yaml` with asymmetric model placement.
- Configured `configs/deepspeed_zero3.yaml` for ZeRO-3 style optimizer sharding.
- Created `pyproject.toml` with dev/eval/deepspeed optional dependencies.

### Observability
- Implemented `src/rlhf_platform/utils/telemetry.py` with:
  - Structured JSON telemetry events (ppo_step, communication, checkpoint, memory)
  - NCCL profiling metrics per rank
  - Distributed metrics collection with rank-aware aggregation

## In Progress / Partially Completed
- Unit tests: scaffolded test files exist; full runtime validation requires `torch` and `pytest` environment installation.
- Multi-node integration: topology and comm hooks are implemented and tested locally; full multi-node validation requires GPU cluster with NCCL.

## Pending / Future Work
- End-to-end multi-node benchmarking and performance profiling on real GPU clusters.
- DeepSpeed / FSDP adapters for tighter ZeRO-3 sharding integration.
- Performance CI (`.github/workflows/ci_perf.yml`) with automated linting, type checking, and benchmarks on PRs.
- More extensive eval harnesses in `evals/` for win-rate distributions and RewardBench metrics.
- Hardware security review and Silent Data Corruption (SDC) detection with checksum-based validations.

---

## Quick Start

### Validate Locally (No GPU)
```bash
# Simulate a 4-rank distributed cluster on CPU
python scripts/simulate_runtime.py --num-ranks 4 --iterations 20

# Check trace logs for deadlocks or synchronization issues
cat /tmp/rlhf_simulate_runtime/trace_rank_0.json
```

### Debug on CPU
```bash
python train.py --use-cpu --num-steps 5 --verbose
```

### Deploy Multi-Node
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<IP> \
  --master_port=29500 \
  train.py \
  --config configs/cluster_topology.yaml \
  --num-steps 1000
```

---

## Key Design Decisions

1. **Asymmetric Topology**: Actor/Critic use FSDP + TP; Reference/Reward use inference replicas. Avoids GPU memory waste and synchronization latency.
2. **Async Rollout Pipeline**: Generator threads populate pinned CPU ring buffer. Training threads pull non-blocking transfers to GPU. Decouples I/O from compute.
3. **Communication Hooks**: Gradient bucket overlapping fires `all_reduce` during backward pass to hide network latency.
4. **Non-Blocking Checkpoints**: Async writer thread streams weights to disk without stalling training loop.
5. **Numerical Safety**: KL clipping, gradient clipping, and NaN detection embedded in hooks to catch Silent Data Corruption early.
