#!/bin/bash
#SBATCH -p a100_short,radiology,gpu4_short,gpu8_short,gpu4_medium,gpu8_medium
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ap-test
#SBATCH --output=logs/eval-%J.log

# Usage (from the repo root):  sbatch gpujob_test.slurm <datalist.csv> [prediction_name] [output_dir]
DATALIST=${1:?usage: sbatch gpujob_test.slurm <datalist.csv> [prediction_name] [output_dir]}
PREDICTION_NAME=${2:-predictions}
OUTPUT_DIR=${3:-output}

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc
conda activate aps

python evaluation.py --datalist_path $DATALIST \
    --models_file configs/ensemble.yaml \
    --output_path $OUTPUT_DIR \
    --prediction_name $PREDICTION_NAME \
    --phase evaluate \
    --num_workers 8
