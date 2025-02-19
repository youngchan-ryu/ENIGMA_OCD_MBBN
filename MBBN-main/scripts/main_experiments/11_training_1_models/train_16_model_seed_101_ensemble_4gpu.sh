#!/bin/bash

EXP_NAME="train_16_model_seed_101_ensemble_4gpu"

module load conda
module load cudnn/9.1.0
module load nccl/2.21.5

conda activate mbbn-env

cd /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main

python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 0 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name ${EXP_NAME} --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 250 --num_heads 4 \
--UQ --UQ_method ensemble --num_ensemble_models 16 --ensemble_models_per_gpu 4 --num_UQ_gpus 4  \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/scripts/main_experiments/11_training_1_models/${EXP_NAME}_error.log \
> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/scripts/main_experiments/11_training_1_models/${EXP_NAME}_output.log