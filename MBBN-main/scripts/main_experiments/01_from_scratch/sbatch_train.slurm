#!/bin/bash
#SBATCH -A m4750
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=pakmasha99@gmail.com
#SBATCH --output=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/output_log/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/slurm/error_log/slurm_%j.err

cd /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5
conda init bash
source mbbn/bin/activate

export SLURM_CPU_BIND="cores"

python3 main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 16 --lr_init_phase2 1.8e-5 --lr_policy_phase2 SGDR \
--workers_phase2 8 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 1.3 \
--exp_name vmd_four_ch_700_tuned_seed3 --seed 3 --sequence_length_phase2 700 \
--intermediate_vec 316 --nEpochs_phase2 300 --num_heads 4 \
--lr_warmup_phase2 900 --lr_gamma_phase2 0.93 --weight_decay_phase2 0.0005 --lr_step_phase2 4000 \
2> /pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error.log
