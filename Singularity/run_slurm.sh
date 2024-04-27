#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/pytorch-nlp-%J.out
#SBATCH --error=logs/pytorch-nlp-%J.err
#SBATCH --job-name="PyTorch NLP"
#SBATCH --partition=gpu

# Replace './containers/project.sif' with the path to your Singularity container if it's different
# Replace '/d/hpc/home/zp68409/Test.py' with the path to your Python script if it's different
srun singularity exec --nv ./containers/project.sif python /d/hpc/home/zp68409/Test.py