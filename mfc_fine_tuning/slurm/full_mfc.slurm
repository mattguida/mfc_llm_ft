#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:2
#SBATCH --qos=gpgpudeeplearn
#SBATCH --ntasks=1
#SBATCH --job-name="ds_full_mfc_finetune"
#SBATCH --cpus-per-task=4
#SBATCH -A punim0478
#SBATCH --mem=64G
#SBATCH -o "deepseek-qw/out/slurm-%J.out" #STDOUT
#SBATCH -e "deepseek-qw/err/slurm-%J.err" #STDERR

module purge
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0

conda activate /data/gpfs/projects/punim0478/guida/unsloth_env

python3 /data/gpfs/projects/punim0478/guida/mfc_fine_tuning/code/full_fine_tune.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

