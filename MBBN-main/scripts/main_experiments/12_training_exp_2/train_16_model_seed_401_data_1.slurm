#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu&hbm80g
#SBATCH -q preempt
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=youngchanryu@snu.ac.kr

EXP_NAME="train_16_model_seed_401_data_1"

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda init bash
source activate mbbn-env

export SLURM_CPU_BIND="cores"

cd /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/scripts/main_experiments/12_training_exp_2/

srun ${EXP_NAME}.sh