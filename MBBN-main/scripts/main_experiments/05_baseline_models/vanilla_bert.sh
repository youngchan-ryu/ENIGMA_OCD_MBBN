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
# ./scripts/main_experiments/05_baseline_models/vanilla_bert.sh
# ./scripts/main_experiments/05_baseline_models/vanilla_bert.sh | tee /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/output.log

## ENIGMA-OCD
python main.py --dataset_name ENIGMA_OCD --base_path /global/homes/p/pakmasha/model/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--task_phase1 vanilla_BERT --step 1 --batch_size_phase1 8 --lr_init_phase1 3e-5 --lr_policy_phase1 step \
--workers_phase1 8 --fine_tune_task binary_classification --target OCD \
--fmri_type timeseries --transformer_hidden_layers 8 --seq_part head  \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name vanilla_bert_seed3 --seed 3 --sequence_length_phase1 100 \
--intermediate_vec 316 --nEpochs_phase1 40 --num_heads 4 \
2> /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/enigma_ocd_error.log