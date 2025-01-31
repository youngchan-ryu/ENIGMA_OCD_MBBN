#!/bin/bash
#SBATCH --job-name=train_mbbn_from_scratch_enigma_ocd
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=4
#SBATCH --nodelist=node3
#SBATCH --qos debug
#SBATCH --time 2:00:00

export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=4
export OMP_NUM_THREADS=10

basedir="/scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/"
cd $basedir

# Prepare for experiments
TZ='Asia/Seoul'; export TZ
start=$(date +%s)
pwd; hostname; date

conda init bash
source activate mbbn-env

srun --ntasks=4 --gpus-per-task=1 python -m torch.distributed.run --nproc_per_node=4 \ 
main.py --dataset_name ENIGMA_OCD --base_path /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main --enigma_path /scratch/connectome/ycryu/MBBN_data \
--step 2 --batch_size_phase2 8 --lr_init_phase2 3e-5 --lr_policy_phase2 step \
--workers_phase2 4 --fine_tune_task binary_classification --target OCD \
--fmri_type divided_timeseries --transformer_hidden_layers 8 \
--divide_by_lorentzian --seq_part head --use_raw_knee --fmri_dividing_type three_channels --use_high_freq \
--spatiotemporal --spat_diff_loss_type minus_log --spatial_loss_factor 4.0 \
--exp_name from_scratch_seed101 --seed 101 --sequence_length_phase2 100 \
--intermediate_vec 316 --nEpochs_phase2 50 --num_heads 4 \
2> /scratch/connectome/ycryu/ENIGMA_OCD_MBBN/MBBN-main/failed_experiments/enigma_ocd_error_from_scratch_seed101.log


# Print time spent & end date
end=$(date +%s)
echo experiment ended
date

awk -v start=$start -v end=$end 'BEGIN{print "Spent "(end-start)/86400" days, "(end-start)/3600" hours"}'

exit 0