#!/bin/bash
#SBATCH --partition=feit-gpu-a100
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --qos=feit
#SBATCH --job-name="multi_cl_cross"
#SBATCH --cpus-per-task=2
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "cross_topic/out/cross_topic_all.out"
#SBATCH -e "cross_topic/err/cross_topic_all.err"

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_BASENAME=$(basename "$MODEL_NAME")

OUTPUT_DIR="/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/cross_topic/${MODEL_BASENAME}_all_topics"
mkdir -p "$OUTPUT_DIR"

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/code/multi_class_different_test_set.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "/data/gpfs/projects/punim0478/guida/mfc_fine_tuning/data/mfc_processed_with_comments_and_15.json"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s