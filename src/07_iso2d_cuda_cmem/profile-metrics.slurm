#!/bin/bash

#SBATCH --job-name=07_iso2d_cuda_cmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000m
#SBATCH --time=00:01:00
#SBATCH --export=ALL
#SBATCH --partition=commons

# cd $HOME/hpc-bootcamp/src/07_iso2d_cuda_cmem
# make clean; make

METRICS="ipc"
METRICS="$METRICS,sm_efficiency"
METRICS="$METRICS,alu_fu_utilization"
METRICS="$METRICS,gld_transactions"
METRICS="$METRICS,gst_transactions"
METRICS="$METRICS,gld_transactions_per_request"
METRICS="$METRICS,gst_transactions_per_request"
METRICS="$METRICS,gld_efficiency"
METRICS="$METRICS,gst_efficiency"

nvprof --metrics $METRICS ./iso2d_cuda -x 1024 -y 1024 -i 800 -t
