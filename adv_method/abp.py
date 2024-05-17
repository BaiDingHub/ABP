import numpy as np

import torch
import os
import pickle
import utils as util
from adv_method.base_method import BaseMethod

from collections import defaultdict
import adv_method
import json
from tqdm import tqdm
import random

class ABP(BaseMethod):
    """  ABP attack method.  """

    def __init__(self, model, use_gpu = False, device_id = [0], synonym_num = 50, embedding_path='', cos_path = '', DEFAULT_ATTACK = 'pwws', wordnet_dir = 'bert-mr', METHOD = 'basic', max_perturbed_percent = 0.25, sample_num = 10000):
        super(ABP,self).__init__(model = model, use_gpu= use_gpu, device_id= device_id)
        self.predictor = self.model.text_pred

        self.synonym_num = synonym_num
        self.embedding_path = embedding_path
        self.cos_path = cos_path

        self.DEFAULT_ATTACK = DEFAULT_ATTACK                # the synonym substitution based attack for the generation of ABP
        self.METHOD = METHOD                                # pick the method (i.e. 'free' or 'search' for ABP_free and ABP_search attacks, respectively)
        self.max_perturbed_percent = max_perturbed_percent  # the max perturbation percent of ABP_free
        self.sample_num = sample_num                        # the sample number for the generation of ABP

        self.wordnet_dir = wordnet_dir
    
    def generate(self, texts, labels):

        if self.DEFAULT_ATTACK == 'pwws':
            adversary = adv_method.PWWS(self.model, self.use_gpu, self.device_ids, self.synonym_num , self.embedding_path, self.cos_path)

        # Initialize the accumulated output difference dict with all 0s
        Influence = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Initialize the accumulated word frequency dict with all 0s
        Preference = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        count = 0
        for text, label in tqdm(zip(texts, labels), total=min(len(texts), self.sample_num)):
            if count > self.sample_num:
                break
            count += 1
        
            adv_sentence = text.copy()
            ori_logit, adv_label = self.query_model(adv_sentence)

        
            if adv_label == label:
                for word in text:
                    Preference[str(label)][word][0] += 1
                log = adversary.attack(text, label)
                success = log['status']
                if success:
                    adv_sentence = log['adv_sentence'].split()
                    adv_label = log['adv_label']
                    text = np.array(text)
                    adv_sentence = np.array(adv_sentence)
                    change_flag = (text != adv_sentence)

                    change_text_list = []
                    index_list = []
                    for index in range(len(change_flag)):
                        if not change_flag[index]:
                            continue
                            
                        change_text = text.copy()
                        change_text[index] = adv_sentence[index]
                        change_text_list.append(change_text)
                        index_list.append(index)
                    
                    adv_logits = self.query_model_group(change_text_list)
                    for i, index in enumerate(index_list):
                        adv_logit = adv_logits[i]
                        word_value = ori_logit[label] - adv_logit[label]
                        Influence[str(label)][text[index]][adv_sentence[index]] += word_value
                        Preference[str(label)][text[index]][1] += 1

        if not os.path.exists('result/abp/{}-{}'.format(self.wordnet_dir,self.DEFAULT_ATTACK)):
            os.makedirs('result/abp/{}-{}'.format(self.wordnet_dir,self.DEFAULT_ATTACK))
            
        with open('result/abp/{}-{}/Influence.json'.format(self.wordnet_dir, self.DEFAULT_ATTACK), 'w', encoding='utf-8') as f:
            json.dump(Influence, f)
        with open('result/abp/{}-{}/Preference.json'.format(self.wordnet_dir, self.DEFAULT_ATTACK), 'w', encoding='utf-8') as f:
            json.dump(Preference, f)
        
        print('The ABP genreation succeeds.')

    
    
    def attack(self, x, y):
        if os.path.exists('result/abp/{}-{}/Influence.json'.format(self.wordnet_dir, self.DEFAULT_ATTACK)):
            with open('result/abp/{}-{}/Influence.json'.format(self.wordnet_dir, self.DEFAULT_ATTACK), 'r', encoding='utf-8') as f:
                self.Influence = json.load(f)
            with open('result/abp/{}-{}/Preference.json'.format(self.wordnet_dir, self.DEFAULT_ATTACK), 'r', encoding='utf-8') as f:
                self.Preference = json.load(f)
        else:
            print('Please Generate ABP Firstly using abp_generate.py file.')

        log = self.abp_attack(x, y)
        
        return log

    def abp_attack(self, ori_sentence, ori_label):
        log ={}
        query_number = 0

        clean_tokens = ori_sentence.copy()
        _, prediction = self.query_model(clean_tokens)
        if prediction != ori_label:
            log['classification'] = False
            return log
        
        log['classification'] = True
        subsitute_list = []
        candidiate_dict = defaultdict()

        for i, token in enumerate(ori_sentence):
            pos_ls = util.get_pos([token])
            pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
            if pos_ls[0] not in  pos_pref:
                continue
            
            ABP_Influence = self.Influence[str(ori_label)]
            ABP_Preference = self.Preference[str(ori_label)]

            if token in ABP_Influence.keys() and token in ABP_Preference.keys():
                # pick the candidate word with the highest value for each word
                candidate_word = token
                candidate_value = 0

                candidate_prob = []
                candidate_words = []
                for c_word in ABP_Influence[token].keys():
                    candidate_value_f = ABP_Influence[token][c_word] / ABP_Preference[token][0]
                    if candidate_value_f > candidate_value:
                        candidate_word = c_word
                        candidate_value = candidate_value_f
                    
                    if candidate_value_f>=0:
                        candidate_prob.append(candidate_value_f)
                    else:
                        candidate_prob.append(0)
                    candidate_words.append(c_word)

                candidate_prob = np.array(candidate_prob)
                if sum(candidate_prob) > 0:
                    candidate_prob = candidate_prob/sum(candidate_prob)
                candidiate_dict[i] = [candidate_prob, candidate_words]

                subsitute_list.append((i, candidate_word, candidate_value))
    
        subsitute_list = sorted(subsitute_list, key = lambda x: x[2],reverse=True)

        if self.METHOD == 'free':
            for i, (id, candidate_word, candidate_value) in enumerate(subsitute_list):
                if (i+1)/len(ori_sentence) < self.max_perturbed_percent:
                    clean_tokens[id] = candidate_word
                    log['perturbation_rate'] = (i+1)/len(ori_sentence)
            _, adv_prediction = self.query_model(clean_tokens)

            log['query_number'] = query_number
            if adv_prediction != ori_label:
                log['status'] = True
            else:
                log['status'] = False

            log['adv_sentence'] = ' '.join(clean_tokens)
            log['adv_label'] = adv_prediction

            return log
    
        elif self.METHOD == 'guide':
            # search attack
            util.setup_seed(2023)
            position_prob = [0] * len(ori_sentence)
            for id, candidate_word, candidate_value in subsitute_list:
                if candidate_value > 0:
                    position_prob[id] = candidate_value
            position_prob = np.array(position_prob)
            if sum(position_prob) == 0:
                log['status'] = False
                log['query_number'] = 0
                return log
            position_prob = position_prob/sum(position_prob)
            adv_log = self.search_attack(ori_sentence, ori_label, candidiate_dict, position_prob, subsitute_list)
            log.update(adv_log)
            return log
        

    def search_attack(self, ori_sentence, ori_label, candidiate_dict, position_prob, subsitute_list):
        log = {}
        
        adv_label = ori_label
        adv_sentence = ori_sentence.copy()
        query_number = 0
        MAX_PERTURB = 2
        N = len(position_prob)
        handle_list = [0] * N

        # greedy search
        for i, (idx, candidate_word, candidate_value) in enumerate(subsitute_list):
                adv_sentence[idx] = candidate_word
                handle_list[idx] = 1
                adv_logit, adv_label = self.query_model(adv_sentence)
                query_number += 1
                if adv_label != ori_label:
                    break
        
        # guided walk
        iter_num = 0
        while(adv_label == ori_label and iter_num < 100):
            perturb_num = np.random.randint(1, MAX_PERTURB+1)
            perturb_indexs = random.choices([i for i in range(N)], position_prob, k=perturb_num)

            for idx in perturb_indexs:
                candidate_prob = candidiate_dict[idx][0]
                candidate_words = candidiate_dict[idx][1]
                candidate = random.choices(candidate_words, candidate_prob)[0]
                adv_sentence[idx] = candidate
                handle_list[idx] = 1
            adv_logit, adv_label = self.query_model(adv_sentence)
            query_number += 1
            iter_num += 1
        
        if adv_label == ori_label:
            log['status'] = False
            log['query_number'] = query_number
            return log
        
        # Prune
        all_perturb_indexs = [i for i in range(len(handle_list)) if handle_list[i] == 1]
        for i in range(20):
            all_perturb_indexs = [i for i in range(len(handle_list)) if handle_list[i] == 1]
            prune_prob = [position_prob[i] for i in range(len(handle_list)) if handle_list[i] == 1]
            prune_prob = np.array(prune_prob)
            prune_prob = 1 - prune_prob/sum(prune_prob)

            all_perturb_num = len(all_perturb_indexs)
            if all_perturb_num == 1:
                break
            prune_num = np.random.randint(1, min(MAX_PERTURB+1, all_perturb_num))
            # prune_num = 1
            perturb_indexs = random.choices(all_perturb_indexs, prune_prob, k=prune_num)
            handle_list_now = handle_list.copy()
            now_adv_sentence = adv_sentence.copy()
            for idx in perturb_indexs:
                now_adv_sentence[idx] = ori_sentence[idx]
                handle_list_now[idx] = 0
            now_adv_logit, now_adv_label = self.query_model(now_adv_sentence)
            if now_adv_label != ori_label:
                adv_sentence = now_adv_sentence
                adv_logit = now_adv_logit
                adv_label = now_adv_label
                handle_list = handle_list_now
            query_number += 1
            
                

        log['status'] = adv_label != ori_label
        log['query_number'] = query_number
        log['adv_sentence'] = ' '.join(adv_sentence)
        log['adv_label'] = adv_label
        diff_count = 0
        for i in range(len(ori_sentence)):
            if ori_sentence[i] != adv_sentence[i]:
                diff_count += 1
        log['perturbation_rate'] = diff_count/len(ori_sentence)
        return log
    

    def query_model(self, text):
        probs = self.predictor([text])[0].data.cpu()
        label = torch.argmax(probs, dim=-1).data.numpy()
        probs = probs.numpy()
        return probs, label

    def query_model_group(self, text_list):
        probs = self.predictor(text_list).data.cpu().numpy()
        return probs