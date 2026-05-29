"""
topology.py: Asymmetric multi-model placement grid for distributed RLHF.

Elite teams at OpenAI, Anthropic, and DeepMind recognize that running Actor, Reference,
Critic, and Reward models with identical distributed strategies creates massive
synchronization bottlenecks. This module implements a hybrid topology:

- Actor & Critic (Active): ZeRO-3 + Tensor Parallelism across nodes
- Reference & Reward (Frozen): Shared inference subnet with replica placement
  and non-gradient tracking, reducing memory footprints and communication overhead.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
import torch
import torch.distributed as dist


class ModelRole(Enum):
    """Role of a model in the RLHF pipeline."""
    ACTOR = "actor"
    CRITIC = "critic"
    REFERENCE = "reference"
    REWARD = "reward"


class ParallelStrategy(Enum):
    """Distributed execution strategy."""
    FSDP_TP = "fsdp_tp"
    INFERENCE = "inference"
    STANDALONE = "standalone"


@dataclass
class ModelPlacement:
    """Placement strategy for a single model within the cluster."""
    role: ModelRole
    model_name: str
    parameter_count: int
    strategy: ParallelStrategy
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    device_list: List[int] = field(default_factory=list)
    precision: str = "fp16"
    offload_to_cpu: bool = False

    def __post_init__(self):
        if self.strategy == ParallelStrategy.FSDP_TP:
            assert self.tensor_parallel_size >= 1, "Tensor parallelism must be >= 1"
            assert self.data_parallel_size >= 1, "Data parallelism must be >= 1"
        if self.role in [ModelRole.REFERENCE, ModelRole.REWARD]:
            assert self.strategy != ParallelStrategy.FSDP_TP, \
                "Frozen models should use INFERENCE or STANDALONE strategy"


@dataclass
class ClusterTopology:
    """Defines the complete asymmetric topology for a multi-node RLHF cluster."""
    world_size: int
    rank: int
    local_rank: int
    num_nodes: int
    gpus_per_node: int

    actor_placement: Optional[ModelPlacement] = None
    critic_placement: Optional[ModelPlacement] = None
    reference_placement: Optional[ModelPlacement] = None
    reward_placement: Optional[ModelPlacement] = None

    actor_critic_group: Optional[dist.ProcessGroup] = None
    reference_reward_group: Optional[dist.ProcessGroup] = None
    actor_group: Optional[dist.ProcessGroup] = None
    critic_group: Optional[dist.ProcessGroup] = None
    reference_group: Optional[dist.ProcessGroup] = None
    reward_group: Optional[dist.ProcessGroup] = None

    def validate(self) -> None:
        """Ensure topology assignments do not conflict."""
        active_ranks = set()
        inference_ranks = set()

        for placement in [self.actor_placement, self.critic_placement]:
            if placement is None:
                continue
            for rank in placement.device_list:
                active_ranks.add(rank)

        for placement in [self.reference_placement, self.reward_placement]:
            if placement is None:
                continue
            for rank in placement.device_list:
                inference_ranks.add(rank)

        shared = active_ranks.intersection(inference_ranks)
        assert not shared, (
            f"Ranks {sorted(shared)} are assigned to both active training and frozen inference groups. "
            "Topology must isolate actor/critic from reference/reward placements."
        )

    def initialize_groups(self) -> None:
        """Create independent communication groups for active and inference subnetworks."""
        assert dist.is_initialized(), "torch.distributed must be initialized before group creation."

        active_ranks = []
        inference_ranks = []

        if self.actor_placement is not None:
            active_ranks.extend(self.actor_placement.device_list)
        if self.critic_placement is not None:
            active_ranks.extend(self.critic_placement.device_list)
        if self.reference_placement is not None:
            inference_ranks.extend(self.reference_placement.device_list)
        if self.reward_placement is not None:
            inference_ranks.extend(self.reward_placement.device_list)

        active_ranks = sorted(set(active_ranks))
        inference_ranks = sorted(set(inference_ranks))
        # Create groups deterministically across all ranks present in the cluster topology.
        # We explicitly call new_group with the same rank lists on all processes to avoid
        # falling back to global-only synchronization.
        try:
            if active_ranks:
                self.actor_critic_group = dist.new_group(ranks=active_ranks)
            if inference_ranks:
                self.reference_reward_group = dist.new_group(ranks=inference_ranks)

            if self.actor_placement is not None:
                self.actor_group = dist.new_group(ranks=sorted(set(self.actor_placement.device_list)))
            if self.critic_placement is not None:
                self.critic_group = dist.new_group(ranks=sorted(set(self.critic_placement.device_list)))
            if self.reference_placement is not None:
                self.reference_group = dist.new_group(ranks=sorted(set(self.reference_placement.device_list)))
            if self.reward_placement is not None:
                self.reward_group = dist.new_group(ranks=sorted(set(self.reward_placement.device_list)))

            # Ensure all processes reach a consistent state before proceeding.
            dist.barrier()
        except Exception as e:
            # If group creation fails, raise a clear error to surface misconfiguration.
            raise RuntimeError(f"Failed to initialize process groups: {e}")

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.local_rank}")

    def is_active_rank(self) -> bool:
        if self.actor_placement and self.rank in self.actor_placement.device_list:
            return True
        if self.critic_placement and self.rank in self.critic_placement.device_list:
            return True
        return False

    def is_inference_rank(self) -> bool:
        if self.reference_placement and self.rank in self.reference_placement.device_list:
            return True
        if self.reward_placement and self.rank in self.reward_placement.device_list:
            return True
        return False

    def to_dict(self) -> Dict:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "actor": self.actor_placement.__dict__ if self.actor_placement else None,
            "critic": self.critic_placement.__dict__ if self.critic_placement else None,
            "reference": self.reference_placement.__dict__ if self.reference_placement else None,
            "reward": self.reward_placement.__dict__ if self.reward_placement else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class TopologyBuilder:
    """Builder pattern for constructing asymmetric cluster topologies."""

    def __init__(self, world_size: int, rank: int, gpus_per_node: int):
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.local_rank = rank % gpus_per_node
        self.num_nodes = world_size // gpus_per_node
        self.topology = ClusterTopology(
            world_size=world_size,
            rank=rank,
            local_rank=self.local_rank,
            num_nodes=self.num_nodes,
            gpus_per_node=gpus_per_node,
        )

    def add_actor(
        self,
        model_name: str,
        param_count: int,
        device_list: Sequence[int],
        tensor_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        if data_parallel_size is None:
            data_parallel_size = max(1, self.world_size // tensor_parallel_size)

        self.topology.actor_placement = ModelPlacement(
            role=ModelRole.ACTOR,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.FSDP_TP,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            device_list=list(device_list),
            precision=precision,
        )
        return self

    def add_critic(
        self,
        model_name: str,
        param_count: int,
        device_list: Sequence[int],
        tensor_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None,
        precision: str = "fp16",
    ) -> "TopologyBuilder":
        if data_parallel_size is None:
            data_parallel_size = max(1, self.world_size // tensor_parallel_size)

        self.topology.critic_placement = ModelPlacement(
            role=ModelRole.CRITIC,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.FSDP_TP,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            device_list=list(device_list),
            precision=precision,
        )
        return self

    def add_reference(
        self,
        model_name: str,
        param_count: int,
        device_list: Sequence[int],
        precision: str = "fp32",
    ) -> "TopologyBuilder":
        self.topology.reference_placement = ModelPlacement(
            role=ModelRole.REFERENCE,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.INFERENCE,
            device_list=list(device_list),
            precision=precision,
        )
        return self

    def add_reward(
        self,
        model_name: str,
        param_count: int,
        device_list: Sequence[int],
        precision: str = "fp32",
    ) -> "TopologyBuilder":
        self.topology.reward_placement = ModelPlacement(
            role=ModelRole.REWARD,
            model_name=model_name,
            parameter_count=param_count,
            strategy=ParallelStrategy.INFERENCE,
            device_list=list(device_list),
            precision=precision,
        )
        return self

    def build(self) -> ClusterTopology:
        self.topology.validate()
        return self.topology


def _resolve_global_ranks(
    device_ids: Sequence[int],
    node_ids: Sequence[int],
    gpus_per_node: int,
) -> List[int]:
    global_ranks = []
    for node_id in node_ids:
        for local_device in device_ids:
            global_ranks.append(node_id * gpus_per_node + local_device)
    return sorted(set(global_ranks))


def _parse_parallel_strategy(strategy_name: str) -> ParallelStrategy:
    normalized = strategy_name.strip().lower()
    if normalized in {"fsdp_tp", "fsdp-tp", "fsdp"}:
        return ParallelStrategy.FSDP_TP
    if normalized in {"inference", "inference_replica", "frozen"}:
        return ParallelStrategy.INFERENCE
    if normalized in {"standalone", "single"}:
        return ParallelStrategy.STANDALONE
    raise ValueError(f"Unknown parallel strategy: {strategy_name}")


def load_topology_from_yaml(
    config_path: str,
    rank: int,
    local_rank: Optional[int] = None,
) -> ClusterTopology:
    """Load cluster topology from YAML and construct communication groups."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cluster = config["cluster"]
    models = config["models"]
    gpus_per_node = int(cluster["gpus_per_node"])
    num_nodes = int(cluster["num_nodes"])
    world_size = int(cluster.get("total_gpus", num_nodes * gpus_per_node))

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % gpus_per_node))

    builder = TopologyBuilder(world_size=world_size, rank=rank, gpus_per_node=gpus_per_node)

    actor_spec = models["actor"]
    actor_strategy = _parse_parallel_strategy(actor_spec.get("parallelism_strategy", "fsdp_tp"))
    if actor_strategy == ParallelStrategy.FSDP_TP:
        builder.add_actor(
            model_name=actor_spec["model_name"],
            param_count=int(actor_spec.get("num_gpus", 0)) * 100_000_000,
            device_list=_resolve_global_ranks(actor_spec["devices"], actor_spec.get("node_ids", [0]), gpus_per_node),
            tensor_parallel_size=int(actor_spec.get("tp_degree", 1)),
            data_parallel_size=int(actor_spec.get("dp_degree", 1)),
            precision=actor_spec.get("precision", "bf16"),
        )
    else:
        builder.topology.actor_placement = ModelPlacement(
            role=ModelRole.ACTOR,
            model_name=actor_spec["model_name"],
            parameter_count=int(actor_spec.get("num_gpus", 0)) * 100_000_000,
            strategy=actor_strategy,
            tensor_parallel_size=int(actor_spec.get("tp_degree", 1)),
            data_parallel_size=int(actor_spec.get("dp_degree", 1)),
            device_list=_resolve_global_ranks(actor_spec["devices"], actor_spec.get("node_ids", [0]), gpus_per_node),
            precision=actor_spec.get("precision", "bf16"),
        )

    critic_spec = models["critic"]
    critic_strategy = _parse_parallel_strategy(critic_spec.get("parallelism_strategy", "fsdp_tp"))
    if critic_strategy == ParallelStrategy.FSDP_TP:
        builder.add_critic(
            model_name=critic_spec["model_name"],
            param_count=int(critic_spec.get("num_gpus", 0)) * 100_000_000,
            device_list=_resolve_global_ranks(critic_spec["devices"], critic_spec.get("node_ids", [0]), gpus_per_node),
            tensor_parallel_size=int(critic_spec.get("tp_degree", 1)),
            data_parallel_size=int(critic_spec.get("dp_degree", 1)),
            precision=critic_spec.get("precision", "bf16"),
        )
    else:
        builder.topology.critic_placement = ModelPlacement(
            role=ModelRole.CRITIC,
            model_name=critic_spec["model_name"],
            parameter_count=int(critic_spec.get("num_gpus", 0)) * 100_000_000,
            strategy=critic_strategy,
            tensor_parallel_size=int(critic_spec.get("tp_degree", 1)),
            data_parallel_size=int(critic_spec.get("dp_degree", 1)),
            device_list=_resolve_global_ranks(critic_spec["devices"], critic_spec.get("node_ids", [0]), gpus_per_node),
            precision=critic_spec.get("precision", "bf16"),
        )

    reference_spec = models["reference"]
    reference_strategy = _parse_parallel_strategy(reference_spec.get("parallelism_strategy", "inference"))
    if reference_strategy == ParallelStrategy.INFERENCE:
        builder.add_reference(
            model_name=reference_spec["model_name"],
            param_count=int(reference_spec.get("num_gpus", 0)) * 100_000_000,
            device_list=_resolve_global_ranks(reference_spec["devices"], reference_spec.get("node_ids", [0]), gpus_per_node),
            precision=reference_spec.get("precision", "fp32"),
        )
    else:
        builder.topology.reference_placement = ModelPlacement(
            role=ModelRole.REFERENCE,
            model_name=reference_spec["model_name"],
            parameter_count=int(reference_spec.get("num_gpus", 0)) * 100_000_000,
            strategy=reference_strategy,
            device_list=_resolve_global_ranks(reference_spec["devices"], reference_spec.get("node_ids", [0]), gpus_per_node),
            precision=reference_spec.get("precision", "fp32"),
        )

    reward_spec = models["reward"]
    reward_strategy = _parse_parallel_strategy(reward_spec.get("parallelism_strategy", "inference"))
    if reward_strategy == ParallelStrategy.INFERENCE:
        builder.add_reward(
            model_name=reward_spec["model_name"],
            param_count=int(reward_spec.get("num_gpus", 0)) * 100_000_000,
            device_list=_resolve_global_ranks(reward_spec["devices"], reward_spec.get("node_ids", [0]), gpus_per_node),
            precision=reward_spec.get("precision", "fp32"),
        )
    else:
        builder.topology.reward_placement = ModelPlacement(
            role=ModelRole.REWARD,
            model_name=reward_spec["model_name"],
            parameter_count=int(reward_spec.get("num_gpus", 0)) * 100_000_000,
            strategy=reward_strategy,
            device_list=_resolve_global_ranks(reward_spec["devices"], reward_spec.get("node_ids", [0]), gpus_per_node),
            precision=reward_spec.get("precision", "fp32"),
        )

    topology = builder.build()
    topology.validate()

    if dist.is_initialized():
        topology.initialize_groups()

    return topology


def create_default_topology(
    world_size: int,
    rank: int,
    gpus_per_node: int,
    actor_model: str = "facebook/opt-1.3b",
    critic_model: str = "facebook/opt-1.3b",
    reference_model: str = "facebook/opt-1.3b",
    reward_model: str = "microsoft/deberta-v3-base",
) -> ClusterTopology:
    builder = TopologyBuilder(world_size, rank, gpus_per_node)

    builder.add_actor(
        model_name=actor_model,
        param_count=1_300_000_000,
        tensor_parallel_size=2,
        data_parallel_size=max(1, world_size // 2),
        device_list=list(range(world_size)),
        precision="fp16",
    )

    builder.add_critic(
        model_name=critic_model,
        param_count=1_300_000_000,
        tensor_parallel_size=2,
        data_parallel_size=max(1, world_size // 2),
        device_list=list(range(world_size)),
        precision="fp16",
    )

    builder.add_reference(
        model_name=reference_model,
        param_count=1_300_000_000,
        device_list=list(range(min(2 * gpus_per_node, world_size))),
        precision="fp32",
    )

    builder.add_reward(
        model_name=reward_model,
        param_count=300_000_000,
        device_list=list(range(min(2 * gpus_per_node, world_size))),
        precision="fp32",
    )

    return builder.build()
