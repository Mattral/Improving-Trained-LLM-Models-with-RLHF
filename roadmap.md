# Roadmap — RLHF Platform Productionization

Reference: original educational walkthrough moved to `docs/educational_deepdive.md`.

## Completed
- Moved original educational README to `docs/educational_deepdive.md` (preserves original context).
- Implemented `train.py` entrypoint wiring topology, model loading, async rollout pipeline, checkpointing, and telemetry.
- Implemented `src/rlhf_platform/alignment/rollout.py` with:
  - `RolloutBuffer` ring buffer using pinned CPU memory and thread-safety
  - `RolloutGenerator` autoregressive generation with top-p sampling
  - `AsyncRolloutPipeline` background generator threads that pin and enqueue rollouts
- Implemented `src/rlhf_platform/distributed/async_io.py` for non-blocking checkpoint writes.
- Implemented `src/rlhf_platform/alignment/ppo_engine.py` core loop and integrated PPO loss primitives.
- Implemented `src/rlhf_platform/utils/telemetry.py` structured telemetry events and NCCL profiling utilities.
- Added topology parsing (`configs/cluster_topology.yaml`) and YAML-driven `topology.py` builder.
- Added unit tests scaffolding for topology and rollout modules (`tests/`).

## In Progress / Partially Completed
- Topology group initialization: explicit `dist.new_group` usage and validation added, but full multi-node runtime validation requires a live multi-node cluster.
- Unit tests: tests for `topology` and `rollout` modules added; running complete test suite requires `torch` and `pytest` in the environment.

## Pending / Future Work
- End-to-end multi-node integration tests and benchmarks (requires access to multi-node GPU cluster and NCCL-enabled environment).
- DeepSpeed / FSDP integration adapters for ZeRO-3 style optimizer sharding and tensor parallelism.
- Performance CI (`.github/workflows/ci_perf.yml`) that runs type checks, linting, and light regression performance tests on PRs.
- More extensive eval harnesses under `evals/` to track win-rate distributions and reward-bench metrics.
- Security review and SDC detection integration (hardware checks + checksum-based validations).

---

If you'd like, I can:
- Add CI workflow and a tiny multi-node emulation harness (single-node multiprocess) to validate group creation logic locally.
- Create a lightweight `docker-compose` / `Makefile` developer flow for local experiments.
