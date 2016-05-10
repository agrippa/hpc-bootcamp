#!/bin/bash

#SBATCH --job-name=01_vecadd-threads
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:01:00
#SBATCH --export=ALL
#SBATCH --partition=commons
#SBATCH --gres=gpu

module load CUDA/7.0.28
cd $HOME/hpc-bootcamp/src/01_vecadd
make clean; make

for THREADS in 32 64 96 128 160 192 224 256 288 320 356 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024; do
    ./vecadd 1048576 $THREADS
done
