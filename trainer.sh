#!/usr/bin/bash
#SBATCH -J trainer
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --constraint=rocky8
#SBATCH --mem 10G
#SBATCH -o trainer.out

source ~/.bashrc
conda activate pytorch

python -u src/price_predictor/train.py --config config.yaml
