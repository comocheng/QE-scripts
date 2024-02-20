#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --job-name=sella_run
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --ntasks=24
#SBATCH --error=neb.err
#SBATCH --output=neb.out
#SBATCH --mem=120GB

source /work/westgroup/chao/q-e/env_qe.sh
export ESPRESSO_PSEUDO="/home/xu.chao/pseudo"

source activate pynta_env
python /work/westgroup/chao/surfmaster/run_sella.py -i input.yaml
