## For test
import sys
sys.path.append('/pscratch/sd/y/ycryu/ENIGMA_OCD_MBBN/MBBN-main')

## For main
from utils import *  #including 'init_distributed', 'weight_loader'
from trainer import Trainer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pathlib import Path
from uncertainty import UQWriter

## YC : CHANGED
import torch
import torch.multiprocessing as mp

## For UQTrainer
import os
import dill
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import Subset, DataLoader
from torch import nn

from utils import *
from trainer import Trainer
from loss_writer import Writer
from metrics import Metrics

class UQTrainer(Trainer):

    def __init__(self, sets, **kwargs):
        super().__init__(sets, **kwargs)
        self.writer = UQWriter(sets, self.val_threshold, **kwargs)
        print(f"model_idx: {self.model_idx}")
    
    ## YC : CHANGED
    def save_checkpoint_(self, epoch, batch_idx, scaler):
        model_idx = self.model_idx

        loss = self.get_last_loss()
        #accuracy = self.get_last_AUROC()
        val_ACC = self.get_last_ACC()
        val_best_ACC = self.get_last_best_ACC()
        val_AUROC = self.get_last_AUROC()
        val_MAE = self.get_last_MAE()
        val_threshold = self.get_last_val_threshold()

        if self.UQ_method == 'ensemble':
            if model_idx is None:
                raise ValueError("model_idx must be provided for ensemble method.")
                
        title = str(self.writer.experiment_title) + '_epoch_' + str(int(epoch))
        directory = self.writer.experiment_folder

        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.amp:
            amp_state = scaler.state_dict()

        # Build checkpoint dict to save.
        ckpt_dict = {
            # 'model_state_dict':self.model.module.state_dict(),  # Distributed case
            'model_state_dict':self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict() if self.optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss,
            'amp_state': amp_state}

        # if val_ACC is not None:
        #     ckpt_dict['val_ACC'] = val_ACC
        if val_AUROC is not None:
            ckpt_dict['val_AUROC'] = val_AUROC
        if val_threshold is not None:
            ckpt_dict['val_threshold'] = val_threshold
        if val_MAE is not None:
            ckpt_dict['val_MAE'] = val_MAE
        if self.lr_handler.schedule is not None:
            ckpt_dict['schedule_state_dict'] = self.lr_handler.schedule.state_dict()
            ckpt_dict['lr'] = self.optimizer.param_groups[0]['lr']
            print(f"current_lr:{self.optimizer.param_groups[0]['lr']}")
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # classification
        if val_AUROC is not None:
            if self.best_AUROC < val_AUROC:
                self.best_AUROC = val_AUROC
                name = "{}_BEST_val_AUROC.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with AUROC:{val_AUROC}')

                if self.best_ACC < val_ACC:
                    self.best_ACC = val_ACC
            elif self.best_AUROC >= val_AUROC:
                # If model is not improved in val AUROC, but improved in val ACC.
                if self.best_ACC < val_ACC:
                    self.best_ACC = val_ACC
                    name = "{}_BEST_val_ACC.pth".format(title)
                    torch.save(ckpt_dict, os.path.join(directory, name))
                    print(f'updating best saved model with ACC:{val_ACC}')

        # regression
        elif val_AUROC is None and val_MAE is not None:
            if self.best_MAE > val_MAE:
                self.best_MAE = val_MAE
                name = "{}_BEST_val_MAE.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with MAE: {val_MAE}')
            else:
                pass
                
        else:
            if self.best_loss > loss:
                self.best_loss = loss
                name = "{}_BEST_val_loss.pth".format(title)
                torch.save(ckpt_dict, os.path.join(directory, name))
                print(f'updating best saved model with loss: {loss}')
            else:
                pass

    def set_model_device(self):  # assigns the model to appropriate devices (e.g., GPU or CPU)
        print(f"DEBUG : self.UQ : {self.UQ} / self.UQ_method : {self.UQ_method} / self.model_idx : {self.model_idx} / self.device_id : {self.device_id} / self.distributed : {self.distributed} / self.rank : {self.rank}")
        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            
            ### DEBUG STATEMENT ###
            print(f"self.gpu: {self.gpu}")
            if self.gpu is None:
                print("self.gpu is None")
            #######################
            
            if self.gpu is not None:
                print('id of gpu is:', self.gpu)
                self.device = torch.device('cuda:{}'.format(self.gpu))
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = self.model.module
            else:
                
                ### DEBUG STATEMENT ###
                print("Distributed training without specific GPU assignment")
                #######################
                
                self.device = torch.device("cuda" if self.cuda else "cpu")
                self.model.cuda()
                if 'reconstruction' in self.task.lower():
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                model_without_ddp = self.model.module

        # manual GPU assignment for ensemble model training
        elif self.UQ and self.UQ_method == 'ensemble' and self.model_idx is not None and self.device_id is not None:
            self.gpu = self.device_id
            self.device = torch.device('cuda:{}'.format(self.device_id))

            torch.cuda.set_device(self.gpu)
            self.model = self.model.to(self.device)
            ### DEBUG STATEMENT ###
            print(f"ensemble_set_model_device!")
            print(f"self.gpu: {self.gpu}")
            print(f"self.device: {self.device}")
            print(f"moved model to: {self.device}")
            #######################
        
        else:
            
            ### DEBUG STATEMENT ###
            print("Single GPU or CPU training")
            #######################
            
            self.device = torch.device("cuda" if self.cuda else "cpu")
            
            ### DEBUG STATEMENT ###
            print(f"self.gpu: {self.gpu}")
            print(f"self.device: {self.device}")
            #######################
            
            #self.model = DataParallel(self.model).to(self.device)
            
            ### DEBUG STATEMENT ###
            self.device = torch.device("cuda:0")   # added for debugging
            self.model = self.model.to(self.device)  
            #######################
            
            ### DEBUG STATEMENT ###
            print(f"moved model to: {self.device}")
            #######################

            

    def eval(self,set):
        ## If set == 'MC_dropout', then set dropout to True
        if set not in ['MC_dropout', 'train', 'val', 'test']:
            raise ValueError(f"Invalid set: {set}")
        self.mode = set
        if set == 'MC_dropout':
            for layer in self.model.modules():
                if isinstance(layer, nn.Dropout):
                    print(f"Enabling MC Dropout for layer {layer} - p={layer.p}")
                    layer.train()
        else:
            self.model = self.model.eval()

    def finish_eval(self, set):
        if set not in ['MC_dropout', 'train', 'val', 'test']:
            raise ValueError(f"Invalid set: {set}")
        if set == 'MC_dropout':
            self.model = self.model.eval()

    def concat_batch_results(self, inout_batches: list):
        inout_keys = inout_batches[0].keys()
        concat_inout = dict()
        for inout in inout_batches:
            for key in inout_keys:
                if key not in concat_inout:
                    concat_inout[key] = inout[key]
                else:
                    if isinstance(inout[key], list):
                        concat_inout[key] += inout[key]
                    elif isinstance(inout[key], torch.Tensor):
                        concat_inout[key] = torch.cat((concat_inout[key], inout[key]), dim=0)
                    else:
                        raise ValueError(f"Invalid inout type: {type(inout[key])}")
        
        return concat_inout
    
    ## YC : divided forward_pass into forward_pass and model_forward_pass to get intermediate results
    ## Be aware when implementing it with MC-dropout code! should replace forward_pass into model_forward_pass

    def model_forward_pass(self,input_dict):
        input_dict = {
            k: (
                v.to(self.device) if (self.cuda and torch.is_tensor(v)) else v
            ) for k, v in input_dict.items()
        }
        for k, v in input_dict.items():
            if torch.is_tensor(v):
                if not v.is_contiguous():
                    v = v.contiguous()
        
        if self.task.lower() == 'test':
            if self.fmri_type in ['timeseries', 'frequency', 'time_domain_high', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high']:
                output_dict = self.model(input_dict['fmri_sequence'])
            elif self.fmri_type == 'divided_timeseries':
                if self.fmri_dividing_type == 'two_channels':
                    output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'three_channels':
                    output_dict = self.model(input_dict['fmri_highfreq_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'four_channels':
                    output_dict = self.model(input_dict['fmri_imf1_sequence'], input_dict['fmri_imf2_sequence'], input_dict['fmri_imf3_sequence'], input_dict['fmri_imf4_sequence'])

        
        #### train & valid ####
        else:
            if self.fmri_type in ['timeseries', 'frequency', 'time_domain_high', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high']:
                output_dict = self.model(input_dict['fmri_sequence'])
            elif self.fmri_type == 'divided_timeseries':
                if self.fmri_dividing_type == 'two_channels':
                    output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'three_channels':
                    output_dict = self.model(input_dict['fmri_highfreq_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
                elif self.fmri_dividing_type == 'four_channels':
                    output_dict = self.model(input_dict['fmri_imf1_sequence'], input_dict['fmri_imf2_sequence'], input_dict['fmri_imf3_sequence'], input_dict['fmri_imf4_sequence'])
                    
                    torch.cuda.synchronize()
                                
        return input_dict, output_dict

    def forward_pass(self,input_dict): 
        input_dict, output_dict = self.model_forward_pass(input_dict)
        
        torch.cuda.nvtx.range_push("aggregate_losses")
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        
        torch.cuda.nvtx.range_pop()
        if self.task.lower() in ['vanilla_bert', 'mbbn', 'mbbn_pretraining', 'test']:
            if self.target != 'reconstruction':
                self.compute_accuracy(input_dict, output_dict)
                
        return loss_dict, loss
        

    def eval_single_UQ_epoch(self,set):  # evaluates the model for a single epoch
        loader = self.test_loader
        subset_indices = list(range(len(self.test_loader.dataset))) * self.num_forward_passes
        subset = Subset(self.test_loader.dataset, subset_indices)
        loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)
        subject_names = [data['subject_name'] for data in loader.dataset]

        self.eval(set)
        input_batches = []
        output_batches = []
        with torch.no_grad():
            for batch_idx, input_dict in enumerate(tqdm(loader, position=0, leave=True)):
                with autocast():
                    ## YC : fixed into model_forward_pass
                    input_dict, output_dict = self.model_forward_pass(input_dict)
                    input_batches.append(input_dict)
                    output_batches.append(output_dict)

        self.finish_eval(set)
        return input_batches, output_batches

    def testing(self):  # manages the testing phase of the model
        # options = ['MC_dropout']
        roc_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'roc_curve.png')
        stat_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'statistics.txt')
        samp_stat_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'sample_statistics.txt')
        if os.path.exists(roc_save_path):
            os.remove(roc_save_path)
        if os.path.exists(stat_save_path):
            os.remove(stat_save_path)
        if os.path.exists(samp_stat_save_path):
            os.remove(samp_stat_save_path)

        input_batches, output_batches = self.eval_single_UQ_epoch('MC_dropout')
        inputs = self.concat_batch_results(input_batches)
        outputs = self.concat_batch_results(output_batches)

        self.compute_accuracy(inputs, outputs)
        self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
        self.writer.compute_confidence(self.writer.confidence_list, self.writer.is_correct_list)

def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['HCP1200', 'ABCD', 'ABIDE', 'UKB', 'ENIGMA_OCD'], default="ENIGMA_OCD")
    parser.add_argument('--fmri_type', type=str, choices=['timeseries', 'frequency', 'divided_timeseries', 'time_domain_low', 'time_domain_ultralow', 'time_domain_high', 'frequency_domain_low', 'frequency_domain_ultralow', 'frequency_domain_high'], default="divided_timeseries")
    parser.add_argument('--intermediate_vec', type=int, default=400)
    parser.add_argument('--abcd_path', default='/scratch/connectome/stellasybae/ABCD_ROI/7.ROI') ## labserver
    parser.add_argument('--ukb_path', default='/scratch/connectome/stellasybae/UKB_ROI') ## labserver
    parser.add_argument('--abide_path', default='/scratch/connectome/stellasybae/ABIDE_ROI') ## labserver
    parser.add_argument('--enigma_path', default='/pscratch/sd/p/pakmasha/MBBN_data') ## Perlmutter 
    parser.add_argument('--base_path', default=base_path) # where your main.py, train.py, model.py are in.
    parser.add_argument('--step', default='1', choices=['1','2','3','4'], help='which step you want to run') # YC : Step 1 : vanilla_BERT / Step 2 : MBBN / Step 3 : divfreqBERT_reconstruction / Step 4 : test
    
    
    parser.add_argument('--target', type=str, default='OCD')
    parser.add_argument('--fine_tune_task',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--prepare_visualization', action='store_true')
    parser.add_argument('--weightwatcher', action='store_true')
    parser.add_argument('--weightwatcher_save_dir', default=None)

    
    
    parser.add_argument('--norm_axis', default=1, type=int, choices=[0,1,None])
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))

    parser.add_argument('--transformer_hidden_layers', type=int,default=8)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--init_method', default='file', type=str, choices=['file','env'], help='DDP init method')
    parser.add_argument('--distributed', default=True)

    # AMP configs:
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    
    # Gradient accumulation
    parser.add_argument("--accumulation_steps", default=1, type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')
    
    # Nsight profiling
    parser.add_argument("--profiling", action='store_true')
    
    #wandb related
    parser.add_argument('--wandb_key', default='d0330ca06936eecd637c3470c47af6d33e1cb277', type=str,  help='default: key for ycryu')
    parser.add_argument('--wandb_mode', default='online', type=str,  help='online|offline')
    parser.add_argument('--wandb_entity', default='youngchanryu-seoul-national-university', type=str)
    parser.add_argument('--wandb_project', default='enigma-ocd_mbbn', type=str)

    
    # dividing
    parser.add_argument('--filtering_type', default='Boxcar', choices=['FIR', 'Boxcar'])
    parser.add_argument('--use_high_freq', action='store_true')
    parser.add_argument('--divide_by_lorentzian', action='store_true')
    parser.add_argument('--use_raw_knee', action='store_true')
    parser.add_argument('--seq_part', type=str, default='head')
    parser.add_argument('--fmri_dividing_type', default='three_channels', choices=['two_channels', 'three_channels', 'four_channels'])
    
    # Dropouts
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.3) 

    # Architecture
    parser.add_argument('--num_heads', type=int, default=12,
                        help='number of heads for BERT network (default: 12)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
                        
    
    ## for finetune
    parser.add_argument('--pretrained_model_weights_path', default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_test', action='store_true', help='test phase of finetuning task')
    
    
    ## spatiotemporal
    parser.add_argument('--spatiotemporal', action = 'store_true')
    parser.add_argument('--spat_diff_loss_type', type=str, default='minus_log', choices=['minus_log', 'reciprocal_log', 'exp_minus', 'log_loss', 'exp_whole'])
    parser.add_argument('--spatial_loss_factor', type=float, default=0.1)
    
    ## ablation
    parser.add_argument('--ablation', type=str, choices=['convolution', 'no_high_freq'])
    
    ## YC : Phase means step
    ## phase 1 vanilla BERT
    parser.add_argument('--task_phase1', type=str, default='vanilla_BERT')
    parser.add_argument('--batch_size_phase1', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--validation_frequency_phase1', type=int, default=10000000)
    parser.add_argument('--nEpochs_phase1', type=int, default=2)  # initially, default=100
    parser.add_argument('--optim_phase1', default='AdamW')
    parser.add_argument('--weight_decay_phase1', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase1', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)
    parser.add_argument('--sequence_length_phase1', type=int ,default=300) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase1', type=int,default=4)
    parser.add_argument('--num_heads_2DBert', type=int, default=12)
    
    ## phase 2 MBBN
    parser.add_argument('--task_phase2', type=str, default='MBBN')
    parser.add_argument('--batch_size_phase2', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--nEpochs_phase2', type=int, default=100)  # initially, default=100
    parser.add_argument('--optim_phase2', default='AdamW')
    parser.add_argument('--weight_decay_phase2', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase2', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--sequence_length_phase2', type=int ,default=300) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase2', type=int, default=4)   # default=4
    
    ##phase 3 pretraining
    parser.add_argument('--task_phase3', type=str, default='MBBN_pretraining')
    parser.add_argument('--batch_size_phase3', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples')
    parser.add_argument('--validation_frequency_phase3', type=int, default=10000000)
    parser.add_argument('--nEpochs_phase3', type=int, default=1000)
    parser.add_argument('--optim_phase3', default='AdamW')
    parser.add_argument('--weight_decay_phase3', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase3', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase3', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase3', type=float, default=0.97)
    parser.add_argument('--lr_step_phase3', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase3', type=int, default=500)
    parser.add_argument('--sequence_length_phase3', type=int ,default=300)
    parser.add_argument('--workers_phase3', type=int,default=4)
    parser.add_argument('--use_recon_loss', action='store_true')
    parser.add_argument('--use_mask_loss', action='store_true') 
    parser.add_argument('--use_cont_loss', action='store_true')
    parser.add_argument('--masking_rate', type=float, default=0.1)
    parser.add_argument('--masking_method', type=str, default='spatiotemporal', choices=['temporal', 'spatial', 'spatiotemporal'])
    parser.add_argument('--temporal_masking_type', type=str, default='time_window', choices=['single_point','time_window'])
    parser.add_argument('--temporal_masking_window_size', type=int, default=20)
    parser.add_argument('--window_interval_rate', type=int, default=2)
    parser.add_argument('--spatial_masking_type', type=str, default='random_ROIs', choices=['hub_ROIs', 'random_ROIs'])
    parser.add_argument('--communicability_option', type=str, default='remove_high_comm_node', choices=['remove_high_comm_node', 'remove_low_comm_node'])
    parser.add_argument('--num_hub_ROIs', type=int, default=5)
    parser.add_argument('--num_random_ROIs', type=int, default=5)
    parser.add_argument('--spatiotemporal_masking_type', type=str, default='whole', choices=['whole', 'separate'])
    
    
    ## phase 4 (test)
    parser.add_argument('--task_phase4', type=str, default='test')
    parser.add_argument('--model_weights_path_phase4', default=None)
    parser.add_argument('--batch_size_phase4', type=int, default=4)
    parser.add_argument('--nEpochs_phase4', type=int, default=1)
    parser.add_argument('--optim_phase4', default='AdamW')
    parser.add_argument('--weight_decay_phase4', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase4', default='SGDR', help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase4', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase4', type=float, default=0.9)
    parser.add_argument('--lr_step_phase4', type=int, default=3000)
    parser.add_argument('--lr_warmup_phase4', type=int, default=100)
    parser.add_argument('--sequence_length_phase4', type=int,default=300) # ABCD 348 ABIDE 280 UKB 464
    parser.add_argument('--workers_phase4', type=int, default=4)
                        
    ## Uncertainty Quantification
    ## YC : CHANGED
    parser.add_argument('--UQ', action='store_true')
    parser.add_argument('--UQ_method', type=str, default='none', choices=['MC_dropout', 'ensemble'])
    parser.add_argument('--num_forward_passes', type=int, default=0) # for MC_dropout
    parser.add_argument('--num_ensemble_models', type=int, default=0) # for ensemble, should use same number when training and testing
    parser.add_argument('--ensemble_models_per_gpu', type=int, default=1)
    parser.add_argument('--UQ_model_weights_path', default=None)
    parser.add_argument('--num_UQ_gpus', type=int, default=1)

    args = parser.parse_args()
        
    return args

def setup_folders(base_path): 
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True) 
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return None

def train_single_model(args, loaded_model_weights_path, phase_num, phase_name, model_idx, device_id):
    # Set the current GPU for this process
    torch.cuda.set_device(device_id)
    # Optionally update args with the device info so Trainer uses the correct device.
    args.device = f"cuda:{device_id}"
    args.model_idx = model_idx
    args.device_id = device_id
    print(f"Starting training for model {model_idx} on GPU {device_id}")
    # Call the original run_phase which trains the model.
    model_path = run_phase(args, loaded_model_weights_path, phase_num, phase_name, model_idx, device_id)
    print(f"Completed training for model {model_idx}, saved to {model_path}")

def run_disributed_phase(args,loaded_model_weights_path,phase_num,phase_name):
        # torchrun: sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ: # for torchrun
        args.world_size = int(os.environ["WORLD_SIZE"])
        #print('args.world_size:',args.world_size)
    elif 'SLURM_NTASKS' in os.environ: # for slurm scheduler
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        pass # torch.distributed.launch
        
    args.distributed = args.world_size > 1 # default: world_size = -1 
    
    # num_gpus = torch.cuda.device_count()
    num_gpus = args.num_UQ_gpus
    if args.num_UQ_gpus > torch.cuda.device_count():
        raise ValueError(f"num_UQ_gpus({args.num_UQ_gpus}) > torch.cuda.device_count({torch.cuda.device_count()})")

    ### DEBUG STATEMENT ###
    print(f'world_size: {args.world_size}')
    print(f'distributed: {args.distributed}')
    print(f'num_gpus: {num_gpus}')
    #######################
    
    # Determine how many ensemble models to train concurrently per GPU.
    # For instance, if you want two models per GPU at a time, then:
    models_per_gpu = args.ensemble_models_per_gpu
    concurrent_models = min(num_gpus * models_per_gpu, args.num_ensemble_models)  # e.g. 4 GPUs * 2 = 8 models concurrently.

    # List all ensemble model indices (for example: [0, 1, 2, ..., args.num_ensemble_models-1])
    ensemble_indices = list(range(args.num_ensemble_models))

    ### DEBUG STATEMENT ###
    print(f'models_per_gpu: {models_per_gpu}')
    print(f'concurrent_models: {concurrent_models}')
    print(f'ensemble_indices: {ensemble_indices}')
    #######################

    # Iterate over ensemble indices in batches of 'concurrent_models'
    for batch_start in range(0, args.num_ensemble_models, concurrent_models):
        processes = []
        batch_indices = ensemble_indices[batch_start: batch_start + concurrent_models]
        print(f"#Training batch models: {batch_indices}")
        for slot, model_idx in enumerate(batch_indices):
            # For assignment, rotate across GPUs. Adjust if you want a different scheduling.
            device_id = slot % num_gpus
            # if device_id != 0:
            #     args.wandb_mode = 'disabled'
            print(f"##slot: {slot} / device_id: {device_id} / model_idx: {model_idx}")
            p = mp.Process(
                target=train_single_model,
                args=(args, loaded_model_weights_path, phase_num, phase_name, model_idx, device_id)
            )
            p.start()
            processes.append(p)
        # Wait for this batch of models to finish training.
        for p in processes:
            p.join()
        print(f"Finished training batch models: {batch_indices}")

    # Optionally, you can collect or post-process the model weight paths here.
    print("All ensemble models trained.")

def run_phase(args,loaded_model_weights_path,phase_num,phase_name, model_idx = None, device_id = None):
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name)
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    if model_idx is not None:
        model_experiment_folder = Path(os.path.join(experiment_folder, f'model_{model_idx}'))
        os.makedirs(model_experiment_folder, exist_ok=True)
        args.experiment_folder = model_experiment_folder
    else:
        args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    print(f'saving the results at {args.experiment_folder}')
    
    # save hyperparameters
    args_logger(args)
    
    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(phase_num, vars(args))
    if args.prepare_visualization:
        S = ['train','val']
    else:
        S = ['train','val','test']

    if args.UQ and model_idx is not None and device_id is not None:
        trainer = UQTrainer(sets=S,**kwargs)
        trainer.training()
    else:
        trainer = Trainer(sets=S,**kwargs)
        trainer.training()

    #S = ['train','val']

    if phase_num == '3' and not fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric)) 

    return model_weights_path


## YC : CHANGED
def test(args,phase_num,model_weights_path):
    UQ = args.UQ
    UQ_method = args.UQ_method
    print(f"UQ : {UQ} / UQ_method : {UQ_method}")
    
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) #, datestamp())
    experiment_folder = Path(os.path.join(args.base_path,'tests', experiment_folder))
    os.makedirs(experiment_folder,exist_ok=True)
    
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name

    if UQ:
        S = [UQ_method]
        if UQ_method == 'MC_dropout':
            # YC : Retrieve the last checkpoint from directory
            file_name_and_time_lst = []
            for f_name in os.listdir(model_weights_path):
                if f_name.endswith('.pth'):
                    written_time = os.path.getctime(os.path.join(model_weights_path,f_name))
                    file_name_and_time_lst.append((f_name, written_time))
            # Backward order of file creation time
            sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

            if len(sorted_file_lst) == 0:
                raise Exception('No model weights found')
            loaded_model_weights_path = os.path.join(model_weights_path,sorted_file_lst[0][0])
            setattr(args,'loaded_model_weights_path_phase' + phase_num, loaded_model_weights_path)
            args_logger(args)
            args = sort_args(args.step, vars(args))
            trainer = UQTrainer(sets=S,**args)

    else:
        # YC : Retrieve the most recent checkpoint from directory
        file_name_and_time_lst = []
        for f_name in os.listdir(model_weights_path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(model_weights_path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # Backward order of file creation time
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        if len(sorted_file_lst) == 0:
            raise Exception('No model weights found')
        loaded_model_weights_path = os.path.join(model_weights_path,sorted_file_lst[0][0])
        setattr(args,'loaded_model_weights_path_phase' + phase_num, loaded_model_weights_path)
        S = ['test']
        args_logger(args)
        args = sort_args(args.step, vars(args))
        trainer = Trainer(sets=S,**args)
    
    trainer.testing()
    

## YC : CHANGED
if __name__ == '__main__':
# def main():
    base_path = os.getcwd() 
    setup_folders(base_path) 
    args = get_arguments(base_path)

    # UQ condition check
    if args.UQ:
        if args.UQ_method == 'none':
            raise Exception('UQ method is not specified')
        elif args.UQ_method == 'MC_dropout':
            if args.num_forward_passes == 0:
                raise Exception('num_forward_passes is not specified')
            elif args.num_ensemble_models != 0:
                raise Exception('num_ensemble_models should not be set for MC_dropout')
            if args.step != '4':
                raise Exception('MC_dropout is only available for testing')
        elif args.UQ_method == 'ensemble':
            if args.num_ensemble_models == 0:
                raise Exception('num_ensemble_models is not specified')
            elif args.num_forward_passes != 0:
                raise Exception('num_forward_passes should not be set for ensemble')
        
        print(f'UQ enabled - method : {args.UQ_method} | step : {args.step}')
        if args.UQ_method == 'ensemble':
            print(f'num_ensemble_models : {args.num_ensemble_models}')
            if args.step == '2':
                args.distributed = False
                print('distributed set False due to manual distributed setting in ensemble method')
        elif args.UQ_method == 'MC_dropout':
            print(f'num_forward_passes : {args.num_forward_passes}')

    # DDP initialization
    # if not (args.step == '2' and args.UQ):
    init_distributed(args)
    print(f"DEBUG : args.distributed : {args.distributed} / args.rank : {args.rank} / args.local_rank : {args.local_rank} / args.world_size : {args.world_size} / args.gpu : {args.gpu}")

    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)

    if step == '4' :
        print(f'starting testing')
        phase_num = '4'
        if args.UQ:
            model_weights_path = args.UQ_model_weights_path
        test(args, phase_num, model_weights_path)
    else:
        print(f'starting phase{step}: {task}')
        if args.UQ and args.UQ_method == 'ensemble':
            if args.UQ_model_weights_path is not None:
                model_weights_path = args.UQ_model_weights_path
                print(f'UQ ensemble model weights loaded from {model_weights_path}')    
            mp.set_start_method("spawn", force=True)
            run_disributed_phase(args,model_weights_path,step,task)
        else:
            run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')
