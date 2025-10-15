#!/bin/bash
#SBATCH --job-name=alltoall-torch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# sBATCH --cpus-per-task=4

set -x
ulimit -c 0 

# Set environment variables for distributed launch (shared across all tasks)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export WORLD_LOCAL_SIZE=$SLURM_NTASKS_PER_NODE

export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Enable GDR (GPUDirect RDMA)
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET_GDR_READ=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1  # Disable IB since we're using Slingshot/OFI

# Libfabric HMEM (Heterogeneous Memory) settings - force GDRCopy usage
export FI_HMEM_CUDA_USE_GDRCOPY=1
export FI_MR_CACHE_MONITOR=userfaultfd

# Slingshot CXI fabric settings
export FI_CXI_DISABLE_HOST_REGISTER=0  # Enable memory registration for CXI
export FI_CXI_DEFAULT_VNI=$(id -u)  # Set VNI for user isolation

# maybe remove
export FI_CXI_RNDZV_PROTO=alt_read  # Use RDMA read for rendezvous
export FI_CXI_RX_MATCH_MODE=hybrid

echo "SLURM Configuration:"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  SLURM_NNODES: $SLURM_NNODES"
echo "  SLURM_NTASKS: $SLURM_NTASKS"
echo "  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "  SLURM_NODELIST: $SLURM_NODELIST"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  WORLD_LOCAL_SIZE: $WORLD_LOCAL_SIZE"

srun --environment=pytorch2506 -u bash -lc '
set -x

# NODE_RANK must be set per-task (SLURM_PROCID is different on each node)
export NODE_RANK=$SLURM_PROCID
echo "Task $SLURM_PROCID (NODE_RANK=$NODE_RANK) starting on $(hostname)"

python benchmark.py --dp-size 1 --in-dtype float8_e4m3fn --out-dtype bfloat16
'