
#!/bin/bash
#SBATCH --job-name=Deep_Learning_classification            # Job name
#SBATCH --output=output.log          # Output file name
#SBATCH --error=error.log            # Error file name
#SBATCH --partition=gpu              # Partition or queue (adjust as needed)
#SBATCH --gres=gpu:1                 # GPU resources, requesting 1 GPU
#SBATCH --ntasks=1                   # Number of tasks or processes
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (1 task per GPU node)
#SBATCH --mem=32000                  # Memory requirement in MB (adjust as needed)
#SBATCH --time=10:00                 # Time limit

# Load necessary modules or activate virtual environment
module load cuda
module load pytorch

# Actual command to be executed
srun -p gpu --gres gpu:1 -n 1 -N 1 --pty --mem 10000 -t 10:00 python3 test.py
