#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 2-00:00:00
#SBATCH --mem 67G
#SBATCH --job-name="P3_ddqn"
#SBATCH --output="slurm_ddqn_%j.out"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
cd ~/CS551-F25-jchao1/Project3
# Install matplotlib if not already installed (for plotting)
pip install matplotlib --quiet
python main_ddqn.py --train_dqn --model_path ./models/dqn_ddqn.pth --resume_from_model --resume_steps 2696199

