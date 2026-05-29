# RLHF Platform Execution & Simulation Scripts

This directory contains utilities for running and validating the distributed RLHF platform.

## Files

### `train.py` (Root)
**Main training entrypoint for production distributed RLHF alignment.**

Located at the repository root, `train.py` orchestrates the complete training pipeline:

- **Distributed initialization** with NCCL (GPU) or Gloo (CPU) backend
- **Asymmetric topology loading** from `configs/cluster_topology.yaml`
- **Multi-model coordination** (Actor, Critic, Reference, Reward)
- **Async rollout pipeline** with thread-safe pinned CPU buffers
- **PPO optimization loop** with gradient accumulation
- **Non-blocking checkpointing** and structured telemetry

#### Usage

**Single-node (debugging):**
```bash
python train.py \
  --config configs/cluster_topology.yaml \
  --num-steps 100 \
  --batch-size 2 \
  --use-cpu \
  --verbose
```

**Multi-node (distributed):**
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=<MASTER_IP> \
  --master_port=29500 \
  train.py \
  --config configs/cluster_topology.yaml \
  --num-steps 1000 \
  --checkpoint-dir /shared/checkpoints
```

#### Arguments

- `--config`: Path to topology YAML config (default: `configs/cluster_topology.yaml`)
- `--num-steps`: Number of PPO update steps (default: 200)
- `--batch-size`: Batch size for rollout generation (default: 1)
- `--max-response-length`: Max token length per rollout (default: 64)
- `--checkpoint-dir`: Directory for async checkpoints (default: `checkpoints`)
- `--prompts`: Custom prompt list (optional)
- `--use-cpu`: Force CPU execution (useful for debugging)
- `--verbose`: Enable debug logging
- `--seed`: Random seed for reproducibility (default: 42)

---

### `simulate_runtime.py` (Scripts)
**Local cluster emulation suite for distributed topology validation.**

Simulates a 4-rank distributed cluster on a single machine using `torch.multiprocessing.spawn`. Validates:

- Asymmetric process group creation without deadlocks
- Thread-safe rollout buffer operations with pinned CPU memory
- Non-blocking tensor transfers across ranks
- Communication hook integration in distributed context
- Barrier synchronization and rank ordering

#### Usage

**Basic (4 ranks, 10 iterations):**
```bash
python scripts/simulate_runtime.py
```

**Custom configuration:**
```bash
python scripts/simulate_runtime.py \
  --num-ranks 4 \
  --iterations 20 \
  --output-dir /tmp/rlhf_traces
```

#### Arguments

- `--num-ranks`: Number of simulated ranks (default: 4)
- `--iterations`: Iterations per rank (default: 10)
- `--output-dir`: Output directory for trace logs (default: temp directory)

#### Output

Generates JSON trace files for each rank (e.g., `trace_rank_0.json`, `trace_rank_1.json`, etc.) containing:

- Rank and world size metadata
- Timestamped trace entries for each operation (barrier, buffer ops, tensor transfers)
- Any errors or deadlock indicators

**Example trace:**
```json
{
  "rank": 0,
  "world_size": 4,
  "trace": [
    "[T=0.001] Starting rank worker initialization.",
    "[T=0.002] Initializing distributed process group with gloo backend.",
    "[T=0.015] Distributed process group initialized successfully.",
    "[T=0.016] All ranks synchronized at barrier 1.",
    "[T=0.017] Loading mock asymmetric topology.",
    "[T=0.018] Topology groups initialized successfully.",
    "[T=0.019] All ranks synchronized at barrier 2.",
    "[T=0.020] Iteration 0: Generated 2 rollouts. Buffer size: 2."
  ]
}
```

#### Exit Codes

- `0`: All ranks completed successfully; no deadlocks detected.
- `1`: Simulation failed; rank-local trace logs available in output directory.

---

## Workflow

### Development & Debugging

1. **Validate topology and comm locally** (no GPU required):
   ```bash
   python scripts/simulate_runtime.py --num-ranks 4 --iterations 20
   ```

2. **Test on single GPU** (CPU-compatible):
   ```bash
   python train.py --use-cpu --num-steps 10 --verbose
   ```

3. **Test on single node with multiple GPUs**:
   ```bash
   torchrun --nproc_per_node=4 train.py --num-steps 10
   ```

### Production Deployment

1. Prepare multi-node cluster with synchronized NTP and InfiniBand fabric.
2. Configure `configs/cluster_topology.yaml` with actual hardware layout.
3. Validate NCCL connectivity: `python scripts/simulate_runtime.py --num-ranks <world_size>`
4. Launch full training with `torchrun` (see multi-node example above).

---

## Troubleshooting

### Deadlock During Simulation
- Check trace logs: `cat /tmp/rlhf_traces/trace_rank_*.json`
- Verify barrier ordering in topology module
- Ensure all ranks reach `dist.barrier()` calls consistently

### OOM During Training
- Reduce `--batch-size`
- Lower `--max-response-length`
- Use `--use-cpu` to estimate memory without GPU overhead
- Check `memory_usage` entries in telemetry logs

### NCCL Timeout (Multi-node)
- Verify network connectivity and low-level RDMA fabric
- Increase NCCL timeout: `export NCCL_COMM_TIMEOUT_MS=600000`
- Check firewall rules on port ranges 29500–29600

---

## Architecture Notes

- **Async Rollout Pipeline**: Generator threads populate a pinned-memory CPU ring buffer; training ranks pull non-blocking transfers to GPU.
- **Asymmetric Topology**: Actor/Critic (active) occupy a distinct process group from Reference/Reward (frozen inference).
- **Communication Hooks**: Gradient bucket overlapping fires `all_reduce` ops concurrently with backward computation to hide network latency.
- **Non-blocking Checkpoints**: Async writer thread pins state to CPU memory and streams to disk without blocking the main training loop.
