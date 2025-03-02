#!/bin/bash

# Submit jobs for different seeds
for seed in 100 200 300 400 500; do
    sbatch run_seed_job.sh $seed
    echo "Submitted job for seed $seed"
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done
