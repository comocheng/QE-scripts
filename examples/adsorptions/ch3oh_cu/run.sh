#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --job-name=ch3oh_ads_run
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --ntasks=4
#SBATCH --error=ads.err
#SBATCH --output=ads.out
#SBATCH --mem=20GB

source /work/westgroup/chao/q-e/env_qe.sh
export ESPRESSO_PSEUDO="/home/xu.chao/pseudo"

source activate finetuna_cpu
python /work/westgroup/chao/surfmaster/adsorption.py -i input.yaml
