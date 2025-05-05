#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:2
#SBATCH --qos=gpgpudeeplearn
#SBATCH --ntasks=1
#SBATCH --job-name="holdout-comments"
#SBATCH --cpus-per-task=2
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "small/out/holdout-comments.out" #STDOUT
#SBATCH -e "small/err/holdout-comments.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

MODEL_NAME="unsloth/Qwen2.5-7B-Instruct"
MODEL_BASENAME=$(basename "$MODEL_NAME")  

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/multi_label/${MODEL_BASENAME}_primary_second_holdout_comments"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/code/multi_label_holdout_comments.py \
    --model_name "$MODEL_NAME" \
    --output_dir "${OUTPUT_DIR}/outputs_multi_label_frame_classification_${MODEL_BASENAME}" \
    --save_path "${OUTPUT_DIR}/train_${MODEL_BASENAME}_multi_label_frames" \
    --json_output_file "${OUTPUT_DIR}/output_${MODEL_BASENAME}_multi_label_frame.jsonl"\
    --file_name "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/primary_and_secondary.json" \
    --subset_size 100

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

