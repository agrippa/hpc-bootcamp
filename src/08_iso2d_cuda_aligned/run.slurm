#!/bin/bash

#SBATCH --job-name=08_iso2d_cuda_aligned
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:01:00
#SBATCH --export=ALL
#SBATCH --partition=interactive

cd $HOME/hpc-bootcamp/src/08_iso2d_cuda_aligned
make clean; make

./iso2d_cuda -x 1024 -y 1024 -i 800 -t
