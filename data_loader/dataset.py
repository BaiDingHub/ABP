import numpy as np
import pandas as pd
import pickle
import os
import utils
# from model_loader.bert_package import BertTokenizer
from pytorch_transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset
import torch
import data_loader
import re
import string
import torchvision.transforms as transforms


class MyTrainDataset(Dataset):
    def __init__(self, dataset, data_dir = './data/', mode = None, is_bert=False, max_seq_length= None, tokenizer = None, word2id=None, model_name = 'bert'):
        self.data_dir = data_dir
        self.is_bert = is_bert
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        
        if mode == 'train':
            self.data, self.labels = data_loader.read_train_text(dataset, data_dir, shuffle=True)
        else:
            self.data, self.labels = data_loader.read_test_text(dataset, data_dir, shuffle=True)
        
        self.tokenizer = tokenizer
        self.word2id = word2id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.is_bert:
            input_ids, input_mask, segment_ids = utils.convert_example_to_feature_for_bert(' '.join(self.data[index]), self.max_seq_length, self.tokenizer, self.model_name)
            return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(self.labels[index])
        else:
            input_ids, input_mask = utils.convert_example_to_feature_for_cnn(' '.join(self.data[index]), self.word2id, self.max_seq_length)
            return torch.tensor(input_ids), torch.tensor(self.labels[index])