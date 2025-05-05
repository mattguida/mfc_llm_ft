#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=35:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="15cross"
#SBATCH --cpus-per-task=1
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "roberta_finetune/output/out/crosstopic_15.out" #STDOUT
#SBATCH -e "roberta_finetune/output/err/crosstopic_15.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/with_15"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/code/roberta_crosstopic.py \
    --data_path "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/mfc_processed_with_comments_and_15.json" \
    --output_dir "${OUTPUT_DIR}/output" \
    --model_name "roberta-large" \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --epochs 10 \
    --max_length 512 \
    --seed 42
    
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

