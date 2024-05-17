import os
import json
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader

from config import *
import utils as util
import pickle
import datetime
import time


class NLIAttacker(object):

    def __init__(self, model, config, attack_method):
        self.model = model
        self.config = config
        self.attack_method = attack_method

        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.use_gpu = True

        self.attack_name = self.config.CONFIG['attack_name']


    def start_attack(self, dataloader):
        attack_method = self.config.Switch_Method['method']
        log = {}
        if attack_method == 'One_Sample_Attack':
            index = getattr(self.config, attack_method)['index']
            for i,(premise, hypothese, label) in enumerate(dataloader):
                if (i == index):
                    log = self.one_sample_attack(premise, hypothese, label)
                    break
        elif attack_method == 'Batch_Sample_Attack':
            log = self.batch_sample_attack(dataloader, **getattr(self.config, attack_method))

        return log



    def one_sample_attack(self, premise, hypothese, label):
        log = {}
        attack_log = self.attack_method.attack(premise, hypothese, label)
        log['pre_data'] = 'premise' + ' '.join(premise)+'hypothese:' + ' '.join(hypothese)
        log['pre_label'] = label
        log.update(attack_log)
        return log



    def batch_sample_attack(self, data_loader, batch):  
        log = {}

        ## Record the attack performance
        success = 0                             # The sample number of successful attacks
        classify_true = 0                       # The sample number of successfully classified after attacks
        sample_num = 0                          # The total number of samples

        query_number_list = []                  # The query number of each attack for target model
        perturbation_rate_list = []             # The perturbation rate of the adversarial example after each attack
        process_time_list = []                  # The processing time for each attack


        for i,(premise, hypothese, label) in enumerate(data_loader):
            if i == batch:
                break
            
            starttime = datetime.datetime.now()
            one_log = self.one_sample_attack(premise, hypothese, label)
            endtime = datetime.datetime.now()
            process_time =  (endtime - starttime).seconds
            process_time_list.append(process_time)


            if not one_log['classification']:
                message = 'The {:3}-th sample is not correctly classified'.format(i)
                log['print_{}'.format(i)] = message
                print(message)
                continue

            sample_num += 1

            # Record the query number
            query_time = one_log['query_number']
            query_number_list.append(query_time)


            if(one_log['status']):
                success += 1

                ## Record the perturbation rate
                perturbation_rate = one_log['perturbation_rate']
                perturbation_rate_list.append(perturbation_rate)

                message = 'The {:3}-th sample takes {:3}s, with the perturbation rate: {:.5}, query number: {:4}. Attack succeeds.'.format(i, process_time, perturbation_rate, query_time)
                print(message)
            else:
                classify_true += 1
                message = 'The {:3}-th sample takes {:3}s, Attack fails'.format(i, process_time)
                print(message)
            
            log['print_{}'.format(i)] = message
        
        message = '\nA total of {:4} samples were selected, {:3} samples were correctly classified, {:3} samples were attacked successfully and {:4} samples failed'.format(batch, sample_num, success, sample_num - success)
        print(message)
        log['print_last'] = message


        acc = sample_num/batch                                                      # The classification accuracy of target model
        attack_acc = (classify_true + success - success)/batch              # The classification accuracy of target model after attack
        success_rate = success/sample_num                                           # The attack success rate of attack method
        average_perturbation_rate = np.mean(perturbation_rate_list).item()          # The average perturbation rate of the adversarial example
        
        average_query_number = np.mean(query_number_list).item()                    # The average query number of each attack
        average_process_time = np.mean(process_time_list).item()                    # The average process time of each attack
        

        log['before_attack_acc'] = acc

        log['after_attack_acc'] = attack_acc
        log['success_rate'] = success_rate
        log['mean_perturbation_rate'] = average_perturbation_rate
        
        log['mean_query_number'] = average_query_number
        log['mean_process_time'] = average_process_time


        return log

        