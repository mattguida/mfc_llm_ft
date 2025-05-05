#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --qos=gpgpudeeplearn
#SBATCH --job-name="kfold"
#SBATCH --cpus-per-task=4
#SBATCH -A punim0478
#SBATCH --mem=265G
#SBATCH -o "kfold/out/qwen-kfold.out" #STDOUT
#SBATCH -e "kfold/err/qwen-kfold.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

MODEL_NAME="unsloth/Qwen2.5-7B-Instruct"
MODEL_BASENAME=$(basename "$MODEL_NAME")  

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/kfold/${MODEL_BASENAME}_kfold_sentence"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/code/k_fold_fine_tune.py \
    --model_name "$MODEL_NAME" \
    --output_dir "${OUTPUT_DIR}/kfold_sentence_${MODEL_BASENAME}" \
    --save_path "${OUTPUT_DIR}/train_${MODEL_BASENAME}_kfold_sentence" \
    --json_output_file "${OUTPUT_DIR}/output_${MODEL_BASENAME}_sentence.jsonl" \
    --file_name "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/mfc_sentence_level_secondary.json" 

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

