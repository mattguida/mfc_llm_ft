#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --job-name="ft_rob_inf"
#SBATCH --cpus-per-task=1
#SBATCH -A punim0478
#SBATCH --mem=256G
#SBATCH -o "roberta_finetune/output/out/ft_roberta_inference.out" #STDOUT
#SBATCH -e "roberta_finetune/output/err/ft_roberta_inference.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/roberta_finetune/code/ft_roberta_inferences.py 
    
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

