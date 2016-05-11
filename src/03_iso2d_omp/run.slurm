#!/bin/bash

#SBATCH --job-name=02_iso2d_seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:01:00
#SBATCH --export=ALL
#SBATCH --partition=interactive

cd $HOME/hpc-bootcamp/src/02_iso2d_seq
make clean; make
echo Running Sequential Version
./iso2d_seq -x 1048 -y 1048 -i 800 -t
echo
echo Running Parallel Version
./iso2d_omp -x 1048 -y 1048 -i 800 -t
