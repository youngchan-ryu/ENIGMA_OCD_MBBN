#!/bin/bash

## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this decides splits

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

# cd /global/homes/p/pakmasha/model/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/01_from_scratch/train_MBBN_from_scratch-ENIGMA_OCD.sh
# ./scripts/main_experiments/01_from_scratch/train_MBBN_from_scratch-ENIGMA_OCD.sh | tee /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/output.log
### EDIT!!!### ./scripts/main_experiments/01_from_scratch/train_MBBN_from_scratch-ENIGMA_OCD.sh | tee /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/output_ROI_322_seq_len_100_seed

## ENIGMA-OCD - perlmutter
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data_mini \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 8 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name from_scratch_seed101_test --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 100 --num_heads 4 \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101_test.log

## ENIGMA-OCD - labserver
python main.py --dataset_name ENIGMA_OCD --base_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /scratch/connectome/ycryu/MBBN_data_mini \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 8 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name from_scratch_seed102_1gpu --seed 102 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 5 --num_heads 4 \
2> /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed102_1gpu.log

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 main.py --dataset_name ENIGMA_OCD \
--base_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /scratch/connectome/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 8 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name from_scratch_seed101 --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 100 --num_heads 4 \
--world_size 4 \
2> /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101.log

# using pdb
# python -m pdb ...
# > /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error.log 2>&1

# sbatch scripts/main_experiments/01_from_scratch/train_from_scratch.slurm
# srun --nodes 1 --nodelist=node3 --ntasks=4 --cpus-per-task=4 --mem-per-cpu=3G --gpus=4 --time=2:00:00 --pty bash -i