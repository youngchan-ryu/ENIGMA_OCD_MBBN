#!/bin/bash

######## parameters info ########
## --step : should be set to 4 if it is test - UQ is test either. 
## --lr_warmup_phase4 : may some error occurs if it is big
## --UQ : enable UQ
## --UQ_method : MC_dropout or ensemble
## If UQ_method == MC_dropout, step == 4
## --num_forward_pass : number of forward pass for MC dropout
## --UQ_model_weights_path : retrieve most recent checkpoint from path directory if doing UQ-dropout.
##     If you need to specify a specific checkpoint, please store only the checkpoint file in the directory and specify the directory path.
## If UQ_method == ensemble, step == 2
## --num_ensemble_models : number of ensemble models
## --UQ_model_weights_path : directory of saving checkpoint at training UQ-ensemble.
##     Defaulted by experiment directory.
## --ensemble_models_per_gpu : number of ensemble models trained in one GPU at once. 
##     If the GPU usage is high, reduce this number, if low, increase this number for faster training.
## --num_UQ_gpus : number of GPUs used for training UQ-ensemble.
#################################

######## Explanation ########
## training for ensemble UQ
## not using DDP in python, but manually assigned gpu number by model_idx
## for example, if num_ensemble_models == 8, and ensemble_models_per_gpu == 2, 2 gpus in node, 
## time 1 - gpu 1 : model 0, 1 / gpu 2 : model 2, 3 is trained
## time 2 - gpu 1 : model 4, 5 / gpu 2 : model 6, 7 is trained
## Checkpoints of each model is saved in the experiment directory with model_{model_idx}/ directory.
#############################

######## NOTICE ########
## 1. If you train ensemble UQ model, all models use same dataset split seed as the given seed. 
## For model initialization seed (which is torch.seed) is different for each model.
## model_idx = [0, 1, ..., num_ensemble_models - 1] and seed = seed + model_idx
########################

######## CAVEATS ########
## 1. If there are no split file, training with multiple GPU/processes cause error. 
## First use single model training to generate split files, and then use ensemble training.
## (Recommendation : use single model training until split generation, and then quit then use ensemble training)
## 2. While training multiple models, if workers_phase != 0 it may cause error - which is training is not executed at first batch training.
## Set workers_phase2 to 0 to avoid this error.
##     (May caused by multiprocessing issues)
########################

######## Debugs ########
## 1. TypeError: 'NoneType' object is not subscriptable at File "/pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/trainer.py", line 406, in train_epoch
##   - Delete split, experiment file, and re-run with 1 model to generate split, and then re-run with ensemble training.
##   - If error occurs, delete pycache files in the directory.
## 2. Training would not work properly due to the multiprocessing issues.
##    You should check all num_ensemble_models are initiated properly.
##    If not, you should re-run the training.
##    Training would not properly initiated at some GPU nodes
##    In my case train 1 model at node for test, ensure working properly -> 16 model training at once, with 16 model training all initiated properly.
########################

## ENIGMA-OCD - labserver
python main.py --dataset_name ENIGMA_OCD --base_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /scratch/connectome/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 0 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name ensemble_training_from_scratch_seed101 --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 10 --num_heads 4 \
--UQ --UQ_method ensemble --num_ensemble_models 16 --ensemble_models_per_gpu 2 --num_UQ_gpus 8  \
2> /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101.log

## ENIGMA-OCD - perlmutter
# salloc -A m4750_g -C gpu -q interactive -t 1:00:00 -N 1 --gpus 4
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 0 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name ensemble_training_from_scratch_seed101 --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 10 --num_heads 4 \
--UQ --UQ_method ensemble --num_ensemble_models 4 --ensemble_models_per_gpu 1 --num_UQ_gpus 4 \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_ensemble_from_scratch_seed101.log

## Full version
python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 0 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name ensemble_training_from_scratch_seed101_real --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 10 --num_heads 4 \
--UQ --UQ_method ensemble --num_ensemble_models 16 --ensemble_models_per_gpu 4 --num_UQ_gpus 4  \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101_real_error.log \
> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101_real_output.log

python main.py --dataset_name ENIGMA_OCD --base_path /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /pscratch/sd/y/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 0 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--seq_part head --fmri_dividing_type four_channels \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name ensemble_four_ch_seed11 --seed 11 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 100 --num_heads 4 \
--UQ --UQ_method ensemble --num_ensemble_models 16 --ensemble_models_per_gpu 4 --num_UQ_gpus 4  \
2> /pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_four_ensemble_11.log
