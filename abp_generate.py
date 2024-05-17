from time import process_time_ns
import adv_method
import data_loader
import user
import model_loader
import user as user_module
import config as config_module
import utils as utils_module
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

import argparse
import yaml

def main():
    utils_module.setup_seed(2070)

    ## Required parameters for target model and hyper-parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        default=None,
                        help="The config parameter yaml file contains the parameters of the dataset, target model and attack method",
                        type=str)

    parser.add_argument('--vGPU',
                        nargs='+',
                        type=int,
                        default=None,
                        help="Specify which GPUs to use.")

    args = parser.parse_args()


    ## Universal parameters
    config = config_module.Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        print(args.config)
        with open(args.config, "r") as load_f:
            config_parameter = yaml.safe_load(load_f)
            config.load_parameter(config_parameter)

    if args.vGPU:
        config.GPU['device_id'] = args.vGPU

    ## Configure the GPU
    # use_gpu = config.GPU['use_gpu']
    # if use_gpu:
    #     device = config.GPU['device_id']
    #     os.environ["CUDA_VISIBLE_DEVICES"]= str(device[0])

    ## Prepare the train dataset
    train_texts, train_labels = data_loader.read_train_text(config.TrainDataset['dataset_name'], config.TrainDataset['data_dir'], shuffle=True)
    
    ## Prepare the target model
    model = getattr(model_loader, 'load_' + config.CONFIG['model_name'])(**getattr(config, config.CONFIG['model_name']), config = config)

    ## Prepare the attack method
    attack_parameter = getattr(config, config.CONFIG['attack_name'])
    attack_name = config.CONFIG['attack_name']
    attack_method = getattr(adv_method, attack_name)(model, **config.GPU, **attack_parameter)
    
    ## The generation of attack
    attack_method.generate(train_texts, train_labels)

if __name__ == "__main__":
    main()
