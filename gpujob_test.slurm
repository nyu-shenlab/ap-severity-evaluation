#!/bin/bash
#SBATCH -p a100_short,radiology,gpu4_short,gpu8_short,gpu4_medium,gpu8_medium
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00
#SBATCH --job-name=ap-test
#SBATCH --output=/gpfs/data/shenlab/Yanqi/ap-severity-evaluation/logs/eval-%J.log

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate aps


MODEL_PATH=/gpfs/data/shenlab/Yanqi/ap-severity-project/saves/exp-02262025-nyu/ordinal-dinov2_vitb16/ordinal-dinov2_vitb16-tuning-61089230
DATALIST=/gpfs/data/shenlab/Yanqi/ap-severity-project/data/goulash/df_final.csv

PREDICTION_NAME=test

python evaluation.py --datalist_path $DATALIST \
    --output_path output \
    --prediction_name $PREDICTION_NAME \
    --pretrain_weights $MODEL_PATH/best_val_auc_S.ckpt \
    --config_path $MODEL_PATH/config.yaml \
    --phase evaluate \
    --val_batchsize 1 \
    --num_workers 8 \
    --visualization_ids 20037 70023



