#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="rob15"
#SBATCH --cpus-per-task=1
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "15/output/out/roberta_15_60k.out" #STDOUT
#SBATCH -e "15/output/err/roberta_15_60k.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/with_15_60k"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/code/roberta_ft.py \
    --data_path "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/mfc_processed_with_comments_and_15_60k.json" \
    --output_dir "${OUTPUT_DIR}/output" \
    --model_name "roberta-large" \
    --batch_size 64 \
    --learning_rate 5e-6 \
    --epochs 10 \
    --max_length 70 \
    --seed 42 
     
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

