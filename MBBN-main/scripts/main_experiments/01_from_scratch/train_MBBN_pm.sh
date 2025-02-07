#!/bin/bash

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate mbbn-env

cd /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main

## Set world_size for DDP / workers_phase2
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 4 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name from_scratch_seed101_4gpu_pm --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 100 --num_heads 4 \
--world_size 4 \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101_4gpu_pm.log