#!/bin/bash

## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# divfreqBERT : name of model (step : 2)
# seed1 : seed is set as 1. this decides splits

## 02 environment setting
# conda activate {your environment}
# cd {your directory which contains 'main.py'}

cd /global/homes/p/pakmasha/model/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/01_from_scratch/train_MBBN_from_scratch-ENIGMA_OCD.sh

## ENIGMA-OCD
python main.py --dataset_name ENIGMA_OCD --base_path /global/homes/p/pakmasha/model/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 4 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--exp_name full_data_from_scratch_seed3 --seed 3 --sequence_length_phase2 128 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--intermediate_vec 322 --nEpochs_phase2 300 --num_heads 7 \
2> /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/enigma_ocd_error.log