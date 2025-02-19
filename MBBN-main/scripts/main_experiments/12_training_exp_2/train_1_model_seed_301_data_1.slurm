#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=youngchanryu@snu.ac.kr

EXP_NAME="train_1_model_seed_301_data_1"

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda init bash
source activate mbbn-env

export SLURM_CPU_BIND="cores"

cd /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/scripts/main_experiments/12_training_exp_2/

srun ${EXP_NAME}.sh