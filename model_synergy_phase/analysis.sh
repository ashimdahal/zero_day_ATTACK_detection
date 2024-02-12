#!/bin/bash
#SBATCH --job-name=ML_RF_training            # Job name
#SBATCH --output=output_random_forrest.log          # Output file name
#SBATCH --error=error_rf.log            # Error file name
#SBATCH --partition=himem              # Partition or queue (adjust as needed)
#SBATCH --ntasks=1                   # Number of tasks or processes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (1 task per GPU node)
#SBATCH --mem=32000                  # Memory requirement in MB (adjust as needed)
#SBATCH --time=10:00                 # Time limit

# Load necessary modules or activate virtual environment
module load python/3.8.6

# Actual command to be executed
srun python data_synthesis.py
