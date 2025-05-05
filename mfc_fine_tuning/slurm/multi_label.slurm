#!/bin/bash
#SBATCH --partition=feit-gpu-a100
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --qos=feit
#SBATCH --job-name="multilabel_sent"
#SBATCH --cpus-per-task=1
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "sentence/out/multi_label_sentence_llama.out" #STDOUT
#SBATCH -e "sentence/err/multi_label_sentence_llama.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_BASENAME=$(basename "$MODEL_NAME")  

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/multi_label/${MODEL_BASENAME}_sentence_level"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/code/multi_label.py \
    --model_name "$MODEL_NAME" \
    --output_dir "${OUTPUT_DIR}/outputs_multi_label_frame_classification_${MODEL_BASENAME}" \
    --save_path "${OUTPUT_DIR}/train_${MODEL_BASENAME}_multi_label_1e-5" \
    --json_output_file "${OUTPUT_DIR}/output_${MODEL_BASENAME}_sentence_level_1e-5.jsonl" \
    --file_name "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/mfc_processed_with_comments.json"
    
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

