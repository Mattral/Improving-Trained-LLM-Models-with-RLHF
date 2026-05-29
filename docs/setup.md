# Setup Guide: Industrial Cluster Deployment Runbook

## Executive Summary

This document provides production-ready deployment configurations for multi-node RLHF training. Covers bare-metal provisioning, SLURM batch submission, Kubernetes orchestration, and critical NCCL environment parameter tuning.

---

## Part I: Bare-Metal Cluster Provisioning

### Pre-Deployment Checklist

```
Hardware Requirements:
  ✓ 2+ nodes with 8× H100/A100 each (80 GB PCIe or SXM)
  ✓ 400 Gbps InfiniBand fabric (or 100 Gbps+ RoCE v2)
  ✓ NTP synchronized clocks (skew < 1 ms)
  ✓ Shared filesystem (NFS or fast object storage)
  ✓ Node interconnect MTU ≥ 9000 (jumbo frames enabled)

Software Stack:
  ✓ CUDA 12.0+ (H100: CUDA 12.1+)
  ✓ NCCL 2.18.1+
  ✓ PyTorch 2.0+
  ✓ OpenMPI 4.1+ (if using host MPI launcher)
  ✓ SLURM (if cluster scheduling)

Network Configuration:
  ✓ IPoIB (IP over InfiniBand) configured on all nodes
  ✓ RDMA CM (connection manager) service running
  ✓ Firmware updated (IB HCA firmware must be recent)
  ✓ Subnet Manager running (NCCL requires it)
```

### Step 1: OS-Level Network Configuration

#### Ubuntu 22.04 / 24.04 Setup

```bash
#!/bin/bash
# Configure InfiniBand fabric on compute nodes

# Install IB libraries
sudo apt-get update
sudo apt-get install -y \
  libibverbs-dev \
  librdmacm-dev \
  libibmad-dev \
  ibutils \
  opensm \
  infiniband-diags

# Enable jumbo frames on IPoIB interface
# Assuming ib0 is the IB interface
sudo ip link set dev ib0 mtu 9000

# Persistent MTU configuration (Ubuntu)
echo "iface ib0 inet dhcp
    mtu 9000" | sudo tee /etc/network/interfaces.d/ib0

# Verify IB fabric
ibstatus  # Should show "PortState: Active"
ibping -S  # Should show successful pings

# Test RDMA connectivity (between Node 0 and Node 1)
# On Node 0:
ibping -S 1
# On Node 1:
ibping -G <Node0_GID> -c 5
```

#### CUDA & NCCL Installation

```bash
#!/bin/bash
# Install CUDA 12.1 for H100

# Download CUDA from NVIDIA (for H100, must use 12.1+)
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run

sudo sh cuda_12.1.1_530.30.02_linux.run --silent --driver --toolkit

# Add to PATH
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install NCCL 2.18+ (use NVIDIA's package repo)
# Follow: https://docs.nvidia.com/deeplearning/nccl/install-guide/

# Verify NCCL installation
/usr/local/cuda/bin/nccl-tests/build/all_reduce_perf -b 4M -e 256M -f 2 -g 8
# Should show > 400 Gbps bandwidth on 8 GPUs (H100 with NVLink)
```

### Step 2: PyTorch & Dependency Installation

```bash
#!/bin/bash
# Install PyTorch 2.0+ with NCCL support

conda create -y -n rlhf python=3.11
conda activate rlhf

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RLHF platform dependencies
pip install transformers accelerate numpy pyyaml tensorboard

# Optional: DeepSpeed for advanced optimizations
pip install deepspeed

# Verify PyTorch NCCL
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
print(f'NCCL available: {hasattr(torch.distributed, \"get_backend\")}')
"
```

### Step 3: NCCL Environment Tuning

**Critical environment variables for production:**

```bash
#!/bin/bash
# NCCL Configuration (export before running train.py)

export NCCL_DEBUG=INFO                    # Enable NCCL logging (remove for production)
export NCCL_IB_DISABLE=0                  # Enable InfiniBand (critical!)
export NCCL_IB_RDMA_CM=1                  # Use RDMA connection manager
export NCCL_IB_VERBS_MAX_MR=7             # Max memory regions for RDMA
export NCCL_IB_TC=41                      # Traffic class for InfiniBand QoS
export NCCL_IB_SL=0                       # Service level (0 = default)
export NCCL_COMM_TIMEOUT_MS=600000        # 10 min timeout (production-safe)
export NCCL_BUFFSIZE=33554432             # 32 MB buffer (default for 8 GPUs)
export CUDA_DEVICE_MAX_CONNECTIONS=1      # Prevent oversubscription

# Optional: CPU affinity (bind processes to NUMA nodes)
export NCCL_TOPO_FILE=/opt/nccl_topo.xml  # Custom topology (if needed)

# For RoCE (if using RoCE instead of InfiniBand):
# export NCCL_IB_DISABLE=0
# export NCCL_IB_RDMA_CM=1
# export NCCL_IB_GID_INDEX=3  # Adjust based on your RoCE setup

# Verification: NCCL tests
# /usr/local/cuda/bin/nccl-tests/build/all_reduce_perf -b 4M -e 256M -f 2 -g 8
```

**Variable explanations:**

| Variable | Value | Rationale |
|----------|-------|-----------|
| `NCCL_IB_DISABLE` | `0` | Enable InfiniBand (required for multi-node) |
| `NCCL_IB_RDMA_CM` | `1` | Use RDMA connection manager (avoids IB subnet mgr issues) |
| `NCCL_COMM_TIMEOUT_MS` | `600000` | 10 min timeout (prevents spurious failures during large collective ops) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Prevent GPU oversubscription (one peer stream per GPU) |
| `NCCL_BUFFSIZE` | `33554432` | 32 MB buffer (adequate for ring collectives on 16 GPUs) |

---

## Part II: SLURM Batch Submission

### SLURM Cluster Configuration

```bash
# File: /etc/slurm/slurm.conf (cluster manager sets this)

# Example for 2 nodes × 8 GPUs each:

NodeName=rlhf-node-[0-1] \
  CPUs=128 \
  RealMemory=1000000 \
  GPUs=8 \
  GresTypes=gpu

PartitionName=gpu-large \
  Nodes=rlhf-node-[0-1] \
  Default=YES \
  MaxTime=UNLIMITED \
  State=UP

AccountingStorageType=accounting_storage/none  # or slurmdbd if using DB accounting

# GPU GRES configuration
GresTypes=gpu
NodeName=rlhf-node-[0-1] Gres=gpu:8
```

### SLURM Batch Script (train.py Submission)

```bash
#!/bin/bash
# File: slurm_submit_rlhf.sh

#SBATCH --job-name=rlhf-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-large
#SBATCH --exclusive
#SBATCH --output=rlhf_%j.log
#SBATCH --error=rlhf_%j.err

# Load environment
module load cuda/12.1
module load gcc/11
conda activate rlhf

# NCCL configuration (production)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_RDMA_CM=1
export NCCL_COMM_TIMEOUT_MS=600000
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Verify node allocation
echo "=== SLURM Node Allocation ==="
sinfo -N
scontrol show job $SLURM_JOB_ID | grep NodeList

# Get master node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=29500
WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
RANK=$SLURM_PROCID
LOCAL_RANK=$SLURM_LOCALID

echo "=== Distributed Training Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# Execute training with srun
srun python -u /path/to/train.py \
    --config configs/cluster_topology.yaml \
    --num-steps 10000 \
    --batch-size 2 \
    --checkpoint-dir /shared/checkpoints \
    --verbose

echo "=== Training Complete ==="
```

**Submission:**
```bash
sbatch slurm_submit_rlhf.sh

# Monitor
squeue -j <job_id> -l
tail -f rlhf_<job_id>.log
```

---

## Part III: Kubernetes / KubeFlow Deployment

### KubeFlow PyTorchJob CRD Configuration

```yaml
# File: rlhf-pytorchjob.yaml

apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: rlhf-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0-cuda12.1-cudnn8-devel
            imagePullPolicy: IfNotPresent
            args: ["python", "-u", "/app/train.py",
                   "--config", "/app/configs/cluster_topology.yaml",
                   "--num-steps", "10000",
                   "--checkpoint-dir", "/checkpoints"]
            resources:
              limits:
                nvidia.com/gpu: 8
              requests:
                nvidia.com/gpu: 8
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_IB_RDMA_CM
              value: "1"
            - name: NCCL_COMM_TIMEOUT_MS
              value: "600000"
            - name: CUDA_DEVICE_MAX_CONNECTIONS
              value: "1"
            volumeMounts:
            - name: code
              mountPath: /app
            - name: checkpoints
              mountPath: /checkpoints
          volumes:
          - name: code
            hostPath:
              path: /mnt/nfs/rlhf-repo
              type: Directory
          - name: checkpoints
            hostPath:
              path: /mnt/shared-storage/checkpoints
              type: DirectoryOrCreate
          restartPolicy: OnFailure
          nodeSelector:
            accelerator: nvidia-h100

    Worker:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0-cuda12.1-cudnn8-devel
            imagePullPolicy: IfNotPresent
            args: ["python", "-u", "/app/train.py",
                   "--config", "/app/configs/cluster_topology.yaml",
                   "--num-steps", "10000",
                   "--checkpoint-dir", "/checkpoints"]
            resources:
              limits:
                nvidia.com/gpu: 8
              requests:
                nvidia.com/gpu: 8
            env:
            - name: NCCL_DEBUG
              value: "INFO"
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: NCCL_IB_RDMA_CM
              value: "1"
            - name: NCCL_COMM_TIMEOUT_MS
              value: "600000"
            - name: CUDA_DEVICE_MAX_CONNECTIONS
              value: "1"
            volumeMounts:
            - name: code
              mountPath: /app
            - name: checkpoints
              mountPath: /checkpoints
          volumes:
          - name: code
            hostPath:
              path: /mnt/nfs/rlhf-repo
              type: Directory
          - name: checkpoints
            hostPath:
              path: /mnt/shared-storage/checkpoints
              type: DirectoryOrCreate
          restartPolicy: OnFailure
          nodeSelector:
            accelerator: nvidia-h100

  cleanPodPolicy: Running
  backoffLimit: 3
```

**Deployment:**
```bash
# Deploy
kubectl apply -f rlhf-pytorchjob.yaml

# Monitor
kubectl get pytorchjob -n kubeflow
kubectl logs -f rlhf-training-master-0 -n kubeflow
kubectl describe pytorchjob rlhf-training -n kubeflow

# Access master pod for debugging
kubectl exec -it rlhf-training-master-0 -n kubeflow -- /bin/bash
```

### Docker Image (for KubeFlow / Kubernetes)

```dockerfile
# File: Dockerfile.rlhf

FROM pytorch/pytorch:2.0-cuda12.1-cudnn8-devel

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libibverbs-dev \
    librdmacm-dev \
    opensm \
    infiniband-diags

# Clone RLHF repo
RUN git clone https://github.com/mattral/LLM-Improving-Trained-Models-with-RLHF.git /app

# Install Python dependencies
WORKDIR /app
RUN pip install -r requirements.txt

# Entrypoint
ENTRYPOINT ["python", "-u", "train.py"]
```

**Build & push:**
```bash
docker build -t myregistry/rlhf:latest -f Dockerfile.rlhf .
docker push myregistry/rlhf:latest
```

---

## Part IV: torchrun Launcher (Simple Multi-Node)

### Direct torchrun Execution

```bash
#!/bin/bash
# File: run_distributed_train.sh

# Set cluster topology
export MASTER_ADDR="node0.example.com"  # Master node IP/hostname
export MASTER_PORT=29500
export WORLD_SIZE=16  # 2 nodes × 8 GPUs
export RANK=$1  # Pass rank as argument when running on each node
export LOCAL_RANK=$(( ($RANK) % 8 ))

# NCCL tuning
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_RDMA_CM=1
export NCCL_COMM_TIMEOUT_MS=600000
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run training
python -u train.py \
    --config configs/cluster_topology.yaml \
    --num-steps 10000 \
    --checkpoint-dir /shared/checkpoints \
    --verbose
```

**Execution on each node:**
```bash
# On Node 0 (8 GPUs)
for i in {0..7}; do
  ssh node1 "RANK=$i bash run_distributed_train.sh" &
done
wait

# On Node 1 (8 GPUs)
for i in {8..15}; do
  ssh node0 "RANK=$((i % 8)) bash run_distributed_train.sh" &
done
wait
```

Alternatively, use `torchrun` directly:

```bash
# Run on Node 0
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=node0.example.com \
  --master_port=29500 \
  train.py \
  --config configs/cluster_topology.yaml \
  --num-steps 10000

# Run on Node 1 (with node_rank=1)
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=node0.example.com \
  --master_port=29500 \
  train.py \
  --config configs/cluster_topology.yaml \
  --num-steps 10000
```

---

## Part V: Troubleshooting & Validation

### Connectivity Verification

```bash
#!/bin/bash
# Test InfiniBand connectivity between nodes

# On Node 0:
ibping -S 1
# Expected output: "Completes successfully"

# On Node 1:
# Run "ibnetdiscover" on master to get Node 0 GUID
ibnetdiscover | grep "Node.*ib0"
# Copy Node 0 GUID
ibping -G <Node0_GUID> -c 5

# NCCL diagnostic (if NCCL installed)
/usr/local/cuda/bin/nccl-tests/build/all_reduce_perf \
    -b 4M -e 256M -f 2 -g 8
# Expected: > 400 Gbps per GPU (H100 with NVLink)
```

### NCCL Debugging

```bash
# Enable verbose NCCL logging during training run
export NCCL_DEBUG=TRACE  # Maximum verbosity
export NCCL_DEBUG_SUBSYS=INIT,COLL  # Specific subsystems

# Run training with maximum debug output
python train.py --verbose --config configs/cluster_topology.yaml

# Check for common issues:
# - "unreach" → Network unreachable (firewall/MTU issue)
# - "timeout" → Communication hung (IB fabric issue or large collective)
# - "transport error" → RDMA transport failure (check firmware)
```

### Performance Benchmarking

```bash
#!/bin/bash
# Measure actual training throughput

python scripts/simulate_runtime.py \
    --num-ranks 4 \
    --iterations 50

# For real training, measure PPO steps/second:
# grep "PPO step" rlhf_<job_id>.log | wc -l
# (Divide by elapsed time to get steps/sec)

# Expected performance (16 GPUs, H100):
# - 10–20 PPO steps per minute (depending on batch size, sequence length)
# - GPU utilization: 85–95%
# - Network utilization: 40–60% during all-reduce phases
```

### Emergency Diagnostics

```bash
# If training hangs, collect diagnostics:
python -c "
import torch
import torch.distributed as dist
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPUs:', torch.cuda.device_count())
print('NCCL available:', hasattr(dist, 'get_backend'))
"

# Check NCCL library version
ldconfig -p | grep nccl

# List GPU state (if training process still alive)
nvidia-smi

# Check network connectivity (while training hung)
for i in {0..15}; do
  ssh rlhf-node-$((i / 8)) "hostname; nvidia-smi -L | head -1"
done
```

---

## Part VI: Production Checklist

```
Pre-Deployment:
  ✓ All nodes have synchronized NTP (ntpstat shows drift < 1 ms)
  ✓ CUDA 12.1+ installed on all nodes
  ✓ NCCL 2.18.1+ compiled and verified
  ✓ InfiniBand fabric active (ibstatus shows "PortState: Active")
  ✓ NCCL tests pass: all_reduce_perf > 400 Gbps per GPU
  ✓ Shared filesystem mounted and writable (/shared/checkpoints)
  ✓ SSH keys configured for passwordless node-to-node access

Deployment:
  ✓ NCCL environment variables exported (NCCL_IB_DISABLE=0, etc.)
  ✓ Batch script or KubeFlow job submitted
  ✓ Training logs streaming to shared storage
  ✓ Checkpoint saving working (verify first checkpoint)

Monitoring:
  ✓ GPU utilization > 85% (check nvidia-smi every 30 sec)
  ✓ PPO steps progressing (check logs every 5 min)
  ✓ No NCCL timeouts or transport errors
  ✓ Memory usage stable (not growing unbounded)

Post-Training:
  ✓ Final checkpoint saved and validated (model can load)
  ✓ Training metrics exported to telemetry logs
  ✓ Nodes returned to available pool
```

---

## Conclusion

This runbook provides production-ready deployment templates for:
- Bare-metal cluster provisioning with NCCL tuning
- SLURM batch scheduling and resource allocation
- Kubernetes/KubeFlow orchestration for cloud environments
- Direct torchrun execution for flexible setups

Follow the checklist rigorously to ensure deadlock-free, high-throughput distributed training across multi-node GPU clusters.

