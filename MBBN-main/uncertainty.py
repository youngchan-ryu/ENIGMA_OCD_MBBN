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


class UQWriter(Writer):
    def __init__(self, sets, val_threshold, **kwargs):
        super().__init__(sets, val_threshold, **kwargs)
        self.confidence_list = []
        self.is_correct_list = []
        self.uncertainty_statistics_dict = {}
        self.uncertainty_quantification_stat = {}

    def compute_confidence(self, confidence_list: list, is_correct_list: list):
        num_bins = 10
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)  # Bin edges from 0 to 1
        bin_indices = np.digitize(confidence_list, bin_edges, right=True)
        bin_middlepoint = (bin_edges[1:] + bin_edges[:-1])/2

        bin_confidences = []
        bin_accuracies = []
        bin_gaps = []
        confidence_nparray = np.array(confidence_list)
        is_correct_nparray = np.array(is_correct_list)
        # ECE is weighted average of calibration error in each bin
        # MCE is maximum calibration error in each bin
        cum_ce = 0
        mce = 0

        # organizing bin elements
        for i in range(1, num_bins + 1):
            indices = np.where(bin_indices == i)[0]  # Get indices of elements in the bin
            if len(indices) > 0:
                avg_confidence = np.mean(confidence_nparray[indices])  # Average confidence
                avg_accuracy = np.mean(is_correct_nparray[indices])  # Accuracy as mean of correct labels
                gap = avg_confidence - avg_accuracy  # Gap between confidence and accuracy

                bin_confidences.append(avg_confidence)
                bin_accuracies.append(avg_accuracy)
                bin_gaps.append(gap)
                cum_ce += np.abs(gap) * len(indices)
                mce = max(mce, np.abs(gap))
            else:
                bin_confidences.append(0)
                bin_accuracies.append(0)
                bin_gaps.append(0)
        
        ece = cum_ce / len(confidence_list)


        # FAR95 statistics
        far95, threshold, fpr, tpr = self.metrics.compute_far95(confidence_list, is_correct_list) # Returns None if no threshold found
        if far95 is None:
            print("\nFalse Acceptance Rate at 95% Recall: No threshold found")
        
        # Drawing ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='o', linestyle='-', label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save the ROC plot
        roc_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'roc_curve.png')
        plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save the ECE/MCE/FAR95 statistics
        self.uncertainty_quantification_stat['ece'] = ece
        self.uncertainty_quantification_stat['mce'] = mce
        self.uncertainty_quantification_stat['far95'] = far95
        self.uncertainty_quantification_stat['threshold'] = threshold

        stat_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'statistics.txt')
        with open(stat_save_path, 'w') as f:
            f.write("==========All samples evaluated==========\n")
            f.write(f"Expected Calibration Error: {ece}\n")
            f.write(f"Maximum Calibration Error: {mce}\n")
            if far95 is not None:
                f.write(f"False Acceptance Rate at 95% Recall: {far95} (threshold: {threshold})\n")

        # drawing plot
        bar_width = 0.08  # Width of the bars
        plt.figure(figsize=(8, 6))

        plt.bar(bin_edges[:-1], bin_accuracies, width=bar_width, align='edge', color='blue', edgecolor='black', label="Outputs")
        plt.bar(bin_edges[:-1], bin_gaps, width=bar_width, align='edge', color='pink', alpha=0.7, label="Gap", bottom=bin_accuracies)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")

        plt.text(0.7, 0.1, f'ECE={ece:.4f}', fontsize=14, bbox=dict(facecolor='lightgray', alpha=0.5))

        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        diagram_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'reliability_diagram.png')
        plt.savefig(diagram_save_path, dpi=300, bbox_inches='tight')

    def sample_uncertainty_statistic(self, subj_name: int, subj_dict: dict, subj_truth: int):

        sample_prediction = 1 if subj_dict['score'].mean().item() > 0.5 else 0
        is_correct = 1 if subj_truth == sample_prediction else 0

        # probabilities list for predicted sample 
        # (which 0 or 1, prob of 1 when sample_prediction == 1 and prob of 0 when sample_prediction == 0)
        if sample_prediction == 1:
            sample_pred_probabilities_list = subj_dict['score'].tolist()
        else:
            sample_pred_probabilities_list = (1 - subj_dict['score']).tolist()
        
        # Calculate uncertainty
        mean = torch.mean(torch.tensor(sample_pred_probabilities_list), axis=0)
        variance = torch.var(torch.tensor(sample_pred_probabilities_list), axis=0)

        confidence_intervals_dict = {}
        confidence_levels = [0.9, 0.95]
        for confidence_level in confidence_levels:
            lower_percentile = (1 - confidence_level) / 2 * 100  # 2.5% for 95% CI
            upper_percentile = (1 + confidence_level) / 2 * 100  # 97.5% for 95% CI

            # Compute confidence intervals for each class
            probabilities_list = torch.stack([(1 - subj_dict['score']), subj_dict['score']], dim=1)
            confidence_intervals = np.percentile(probabilities_list, [lower_percentile, upper_percentile], axis=0)
            confidence_intervals_dict[confidence_level] = confidence_intervals

        stat_save_path = os.path.join(self.kwargs.get("experiment_folder"), 'sample_statistics.txt')
        with open(stat_save_path, 'a') as f:
            f.write(f"\nStatistics for sample {subj_name}\n")
            f.write(f"No. of forward passes: {len(sample_pred_probabilities_list)}\n")
            f.write(f"Final prediction: {sample_prediction}\n")
            f.write(f"True label: {subj_truth}\n")
            f.write(f"Correct: {True if is_correct else False}\n")
            f.write(f"Prediction probability: {mean}\n")
            f.write(f"Variance: {variance}\n")
            for confidence_level in confidence_levels:
                for i in range(2):
                    f.write(f"Class {i}: {confidence_level * 100}% CI = [{confidence_intervals_dict[confidence_level][0, i]}, {confidence_intervals_dict[confidence_level][1, i]}]\n")
            f.write("\n")

        self.uncertainty_statistics_dict[subj_name] = {
            'mean': mean,
            'variance': variance,
            'confidence_intervals': confidence_intervals_dict,
            'is_correct': is_correct,
            'sample_prediction': sample_prediction,
            'sample_pred_probabilities_list': sample_pred_probabilities_list,
            'truth': subj_truth
        }

        return mean, is_correct
        ## /YC

    
    def accuracy_summary(self, mid_epoch, mean, std):
        pred_all_sets = {x:[] for x in self.sets}   # dictionary to store predictions
        truth_all_sets = {x:[] for x in self.sets}  # dictionary to store ground truth values
        std_all_sets = {x:[] for x in self.sets}  # dictionary to store prediction errors
        metrics = {}
        confidence_list = []
        is_correct_list = []
        
        for subj_name,subj_dict in self.subject_accuracy.items():  # per-subject prediction scores (score), ground truth labels (truth), and the set (mode) they belong to
            
            if self.fine_tune_task == 'binary_classification':
                subj_dict['score'] = torch.sigmoid(subj_dict['score'].float())

            # subj_dict['score'] denotes the logits for sequences for a subject
            subj_pred = subj_dict['score'].mean().item() 
            subj_error = subj_dict['score'].std().item()

            subj_truth = subj_dict['truth'].item()
            subj_mode = subj_dict['mode'] # train, val, test

            conf, is_corr = self.sample_uncertainty_statistic(subj_name, subj_dict, subj_truth)
            confidence_list.append(conf)
            is_correct_list.append(is_corr)

            # with open(os.path.join(self.per_subject_predictions,'iter_{}.txt'.format(self.eval_iter)),'a+') as f:
            #     f.write('subject:{} ({})\noutputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_pred,subj_error,subj_truth))
            
            pred_all_sets[subj_mode].append(subj_pred) # don't use std in computing AUROC, ACC
            std_all_sets[subj_mode].append(subj_error)
            truth_all_sets[subj_mode].append(subj_truth)

        for (name,pred),(_, std),(_,truth) in zip(pred_all_sets.items(), std_all_sets.items(), truth_all_sets.items()):
            if len(pred) == 0:
                continue

            if self.fine_tune_task == 'regression':
                ## return to original scale ##
                unnormalized_pred = [i * std + mean for i in pred]
                unnormalized_truth = [i * std + mean for i in truth]

                metrics[name + '_MAE'] = self.metrics.MAE(unnormalized_truth,unnormalized_pred)
                metrics[name + '_MSE'] = self.metrics.MSE(unnormalized_truth,unnormalized_pred)
                metrics[name +'_NMSE'] = self.metrics.NMSE(unnormalized_truth,unnormalized_pred)
                metrics[name + '_R2_score'] = self.metrics.R2_score(unnormalized_truth,unnormalized_pred)
                
            else:
                metrics[name + '_Balanced_Accuracy'] = self.metrics.BAC(truth,[x>0.5 for x in torch.Tensor(pred)])
                metrics[name + '_Regular_Accuracy'] = self.metrics.RAC(truth,[x>0.5 for x in torch.Tensor(pred)]) # Stella modified it
                metrics[name + '_AUROC'] = self.metrics.AUROC(truth,pred)             
                metrics[name +'_best_bal_acc'], metrics[name + '_best_threshold'],metrics[name + '_gmean'],metrics[name + '_specificity'],metrics[name + '_sensitivity'],metrics[name + '_f1_score'] = self.metrics.ROC_CURVE(truth,pred,name,self.val_threshold)

            self.current_metrics = metrics
            
            
        for name,value in metrics.items():
            self.scalar_to_tensorboard(name,value)
            if hasattr(self,name):
                l = getattr(self,name)
                l.append(value)
                setattr(self,name,l)
            else:
                setattr(self, name, [value])
                
        self.eval_iter += 1
        if mid_epoch and len(self.subject_accuracy) > 0:
            self.subject_accuracy = {k: v for k, v in self.subject_accuracy.items() if v['mode'] == 'train'}
        else:
            self.subject_accuracy = {}

        self.confidence_list = confidence_list
        self.is_correct_list = is_correct_list


class UQTrainer(Trainer):

    def __init__(self, sets, **kwargs):
        super().__init__(sets, **kwargs)
        self.writer = UQWriter(sets, self.val_threshold, **kwargs)
    
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
    
    def forward_pass(self,input_dict): 
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

    def eval_epoch(self,set):  # evaluates the model for a single epoch
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
                    input_dict, output_dict = self.forward_pass(input_dict)
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

        input_batches, output_batches = self.eval_epoch('MC_dropout')
        inputs = self.concat_batch_results(input_batches)
        outputs = self.concat_batch_results(output_batches)

        self.compute_accuracy(inputs, outputs)
        self.writer.accuracy_summary(mid_epoch=False, mean=None, std=None)
        self.writer.compute_confidence(self.writer.confidence_list, self.writer.is_correct_list)