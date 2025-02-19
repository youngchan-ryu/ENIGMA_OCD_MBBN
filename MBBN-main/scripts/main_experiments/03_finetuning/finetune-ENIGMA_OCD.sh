#!/bin/bash
## 01 scripts explanation
# ABCD : name of dataset
# sex : name of task
# MBBN: name of model (step : 2)
# seed 1 : seed is set as 1. this decides dataset splits

## 02 environment setting
# conda activate {your environment}
# cd /global/homes/p/pakmasha/model/MBBN-main
# salloc -A m4750_g -C gpu -q interactive -t 4:00:00 -N 1 --gpus 1
# source mbbn/bin/activate
# ./scripts/main_experiments/03_finetuning/finetune-ENIGMA_OCD.sh

## 03 experiment
# ENIGMA_OCD
python main.py --dataset_name ENIGMA_OCD --base_path /global/homes/p/pakmasha/model/MBBN-main --enigma_path /pscratch/sd/p/pakmasha/MBBN_data \
--step 2 --batch_size_phase2 32 --lr_init_phase2 3e-5 --workers_phase2 8 \
--fine_tune_task binary_classification --target OCD --intermediate_vec 400 --fmri_type divided_timeseries \
--nEpochs_phase2 100 --filtering_type Boxcar --transformer_hidden_layers 8 --num_heads 4 --exp_name finetune_seed1 \
--seed 1 --sequence_length_phase2 100 --divide_by_lorentzian --seq_part head --use_raw_knee \
--fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log \
--spatial_loss_factor 1.0 --finetune \
--pretrained_model_weights_path /global/homes/p/pakmasha/model/MBBN-main/weights/Schaefer400.pth \
2> /global/homes/p/pakmasha/model/MBBN-main/failed_experiments/enigma_ocd_error.log

#--prepare_visualization


pretraining weight + gaussian small var -> initial


torch.seed(101) -> torch.seed(102) -> 103 ... 116