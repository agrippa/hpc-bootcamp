#!/bin/bash

#SBATCH --job-name=00_vecadd
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
./vecadd 65536
