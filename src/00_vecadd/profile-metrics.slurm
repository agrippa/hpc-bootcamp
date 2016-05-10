#!/bin/bash

#SBATCH --job-name=00_vecadd-profile-metrics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:01:00
#SBATCH --export=ALL
#SBATCH --partition=commons
#SBATCH --gres=gpu

module load CUDA/7.0.28
cd $HOME/hpc-bootcamp/src/00_vecadd
make clean; make
# ipc tells us the number of instructions executed per cycle on the GPU
# sm_efficiency tells us the percentage of time an SM on the GPU is doing useful
#   work.
# alu_fu_utilization indicates how well the CUDA cores are being used
nvprof --metrics ipc,sm_efficiency,alu_fu_utilization ./vecadd 131072
