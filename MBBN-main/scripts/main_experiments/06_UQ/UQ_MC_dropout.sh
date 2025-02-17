#!/bin/bash

## UQ_model_weights_path : retrieve most recent checkpoint from path directory if doing UQ-dropout.
## If you need to specify a specific checkpoint, please store only the checkpoint file in the directory and specify the directory path.

# ## ENIGMA-OCD - labserver
python main.py --dataset_name ENIGMA_OCD --base_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /scratch/connectome/ycryu/MBBN_data_mini \
--step 4 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name test_evaluation_seed101 --seed 101 \
--intermediate_vec 316 --num_heads 4 \
--sequence_length_phase4 100 --lr_warmup_phase4 1 --workers_phase4 1 \
--UQ --UQ_method MC_dropout --num_forward_pass 16 --UQ_model_weights_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/experiments/ENIGMA_OCD_mbbn_OCD_from_scratch_seed101 \
--wandb_mode disabled \
2> /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/uq_mc_dropout_seed101.log

## perlmutter
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 4 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name test_evaluation_seed101 --seed 101 \
--intermediate_vec 316 --num_heads 4 \
--sequence_length_phase4 100 --lr_warmup_phase4 1 --workers_phase4 1 \
--UQ --UQ_method MC_dropout --num_forward_pass 16 --UQ_model_weights_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/experiments/ENIGMA_OCD_mbbn_OCD_from_scratch_seed101_1gpu_perlmutter \
--wandb_mode disabled \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/uq_mc_dropout_seed101.log