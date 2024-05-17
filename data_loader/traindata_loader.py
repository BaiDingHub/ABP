import csv
import logging
import sys
import numpy as np
import os
import random
import string
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import pickle
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, BatchSampler
import re
import pandas as pd


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']

def helper_name(x):
    name = x.split('/')[-1]
    return int(name.split('_')[0])

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_train_text(dataset, data_dir="./data/dataset", shuffle = False):
    print("Reading the dataset: %s" % (dataset))
    label_list = []
    clean_text_list = []
    if dataset == 'ag':
        with open(os.path.join(data_dir, dataset, "train.csv"), "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in csv_reader:
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(clean_str(text.strip()).split())
    elif dataset == 'imdb':
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, dataset + '/train/pos')
        neg_path = os.path.join(data_dir, dataset + '/train/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        clean_text_list = [clean_str(text.strip()).split() for text in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)

    elif dataset == 'mr':
        if not os.path.exists(os.path.join(data_dir, dataset, 'mr_train')):
            text_set = []
            label_set = []
            with open(os.path.join(data_dir, dataset, 'rt-polarity.neg'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(0)

            with open(os.path.join(data_dir, dataset, 'rt-polarity.pos'), 'r', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                for line in lines:
                    text = line.strip(' ').strip('\n')
                    text_set.append(text)
                    label_set.append(1)
            random.seed(0)
            index =list(range(len(text_set)))
            random.shuffle(index)
            text_set = np.array(text_set)[index].tolist()
            label_set = np.array(label_set)[index].tolist()

            with open(os.path.join(data_dir, dataset, 'mr_train'), 'w') as f:
                for text, label in zip(text_set[:-1000], label_set[:-1000]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')
            
            with open(os.path.join(data_dir, dataset, 'mr'), 'w') as f:
                for text, label in zip(text_set[-1000:], label_set[-1000:]):
                    f.write(str(label) + ' '+ text)
                    f.write('\n')

        with open(os.path.join(data_dir, dataset, 'mr_train'), 'r') as f:
            for line in f.readlines():
                label, sep, text = line.partition(' ')
                label_list.append(int(label))
                clean_text_list.append(clean_str(text.strip(' ')).split(' '))  
    elif dataset == 'sst':
        pos_list = []
        neg_list = []
        
        with open(os.path.join(data_dir, dataset, 'train.tsv'), "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines[1:]:
                line = line.split('\t')
                clean_text_list.append(clean_str(line[0].lower().strip(' ')).split(' '))
                label_list.append(int(line[1]))
        if not os.path.exists(os.path.join(data_dir, dataset, 'sst')):
            test_text_list = []
            test_label_list = []
            with open(os.path.join(data_dir, dataset, 'dev.tsv'), "r", encoding="utf-8") as fp:
                lines = fp.readlines()
                for line in lines[1:]:
                    line = line.split('\t')
                    test_text_list.append(clean_str(line[0].lower().strip(' ')).split(' '))
                    test_label_list.append(int(line[1]))

            index =list(range(len(test_text_list)))
            random.shuffle(index)
            test_text_list = np.array(test_text_list)[index].tolist()
            test_label_list = np.array(test_label_list)[index].tolist()

            with open(os.path.join(data_dir, dataset, 'sst'), 'w') as f:
                for text, label in zip(test_text_list[:1000], test_label_list[:1000]):
                    f.write(str(label) + ' '+ ' '.join(text))
                    f.write('\n')
    else:
        raise NotImplementedError
    
    if shuffle:
        index =list(range(len(clean_text_list)))
        random.shuffle(index)
        clean_text_list = np.array(clean_text_list)[index].tolist()
        label_list = np.array(label_list)[index].tolist()
    
    return clean_text_list, label_list



def read_train_nli_text(dataset, data_dir="./data", shuffle = False):
    label_set = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    if dataset == 'snli':
        file_name = os.path.join(data_dir, dataset, 'snli_1.0_dev.txt')
        with open(file_name, 'r') as f:
            rows = [row.split('\t') for row in f.readlines()[1:]]

        premises = [row[5].rstrip().split() for row in rows if row[0] in label_set]
        hypotheses = [row[6].rstrip().split() for row in rows if row[0] in label_set]
        labels = [label_set[row[0]] for row in rows if row[0] in label_set]

        # snli_data = pd.read_json(os.path.join(data_dir, dataset, 'snli_1.0_dev.jsonl'), lines=True)
        # premises = snli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        # hypotheses = snli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        # labels = snli_data["gold_label"].apply(lambda label: label_set[label])

    elif dataset == 'mnli':
        mnli_data = pd.read_json(os.path.join(data_dir, dataset, 'multinli_1.0_train.jsonl'), lines=True)
        premises = mnli_data["sentence1"].apply(lambda premise: premise.rstrip().split())
        hypotheses = mnli_data["sentence2"].apply(lambda hypothese: hypothese.rstrip().split())
        labels = mnli_data["gold_label"].apply(lambda label: label_set[label])
    
    if shuffle:
        index =list(range(len(premises)))
        random.shuffle(index)
        premises = np.array(premises)[index].tolist()
        hypotheses = np.array(hypotheses)[index].tolist()
        labels = np.array(labels)[index].tolist()

    return premises, hypotheses, labels
    
    


def read_text(path, data_dir="./data/"):
    print("reading path: %s" % (data_dir + path))
    label_list = []
    clean_text_list = []
    if (
        path.startswith("ag_news")
        or path.startswith("dbpedia")
        or path.startswith("yahoo")
        or path.startswith("yelp")
    ):
        with open(data_dir + "%s.csv" % path, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            count = 0
            for row in csv_reader:
                count += 1
                label_list.append(int(row[0]) - 1)
                text = " . ".join(row[1:]).lower()
                clean_text_list.append(text_to_tokens(text))
    elif path.startswith('sst-2'):
        pos_list = []
        neg_list = []
        pos_num = 0
        neg_num = 0
        with open(data_dir + "%s.tsv" % path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.split('\t')
                if int(line[1]) == 0:
                    neg_list.append(line[0].lower().strip())
                    neg_num += 1
                else:
                    pos_list.append(line[0].lower().strip())
                    pos_num += 1
        text_list = pos_list + neg_list
        clean_text_list = [text_to_tokens(s) for s in text_list]
        label_list = [1] * pos_num + [0] * neg_num
    elif path.startswith('imdb'):
        pos_list = []
        neg_list = []
        
        pos_path = os.path.join(data_dir, path + '/pos')
        neg_path = os.path.join(data_dir, path + '/neg')
        pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
        neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

        pos_files = sorted(pos_files, key=lambda x : helper_name(x))
        neg_files = sorted(neg_files, key=lambda x : helper_name(x))

        pos_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in pos_files]
        neg_list = [open(x, 'r', encoding='utf-8').read().lower().strip().replace('<br />', ' ') for x in neg_files]
        text_list = pos_list + neg_list
        # clean the texts
        clean_text_list = [text_to_tokens(s) for s in text_list]
        label_list = [1]*len(pos_list) + [0]*len(neg_list)

    elif path.startswith('mr'):
        with open(os.path.join(data_dir, path, 'mr-train'), 'r') as f:
            for line in f:
                label, sep, text = line.partition(' ')
                label_list.append(int(label))
                clean_text_list.append(text[:-1].split())
        
    else:
        raise NotImplementedError
    
    return clean_text_list, label_list

def text_to_tokens(text):
    """
    Clean the raw text.
    """
    toks = word_tokenize(text)
    spliter = ['\'', '#', '!', '\"', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', ':', ';', '<', '=', '>', '?','@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\t', '\n']
    toks = [token for token in filter(lambda x: x not in spliter, toks)]
    return toks