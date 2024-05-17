import numpy as np

import torch
import os
import pickle
import utils as util
from adv_method.base_method import BaseMethod

class PWWS(BaseMethod):
    def __init__(self, model, use_gpu = False, device_id = [0], synonym_num = 30, embedding_path='', cos_path = ''):
        super(PWWS,self).__init__(model = model, use_gpu= use_gpu, device_id= device_id)
        self.predictor = self.model.text_pred
        self.synonym_dict = {}

        self.synonym_num = synonym_num
        self.embedding_path = embedding_path
        self.cos_path = cos_path

        self.idx2word, self.word2idx = util.load_embedding_dict_info(embedding_path)
        self.cos_sim = util.load_cos_sim_matrix(cos_path)

    def attack(self, x, y):
        
        log = self.pwws_attack(x, y)
        
        return log


    def pwws_attack(self, ori_sentence, ori_label):

        log ={}
        query_number = 0


        clean_tokens = ori_sentence
        _, prediction = self.query_model(clean_tokens)
        query_number += 1
        if prediction != ori_label:
            log['classification'] = False
            return log
        
        log['classification'] = True

        adv_sentence = ori_sentence
        adv_label = ori_label
        perturbed_tokens = clean_tokens.copy()
        success = False

        replace_list, new_query = self.repalce_order_decision(clean_tokens, ori_label)
        query_number += new_query
        for i in range(len(replace_list)):
            tmp = replace_list[i]
            perturbed_tokens[tmp[0]] = tmp[1]
            _, prediction = self.query_model(perturbed_tokens)
            query_number += 1
            
            if int(prediction) != int(ori_label):
                success = True
                adv_label = prediction
                adv_sentence = perturbed_tokens.copy()
                break

        log['status'] = success
        log['query_number'] = query_number
        if not success:
            return log


        perturbation_rate = self.check_diff(ori_sentence, adv_sentence)/len(clean_tokens)
        # print(adv_sentence)
        log['adv_sentence'] = ' '.join(adv_sentence)
        log['perturbation_rate'] = perturbation_rate
        log['adv_label'] = adv_label
        return log

    def repalce_order_decision(self, clean_tokens, ori_label):
        """[Decision the replace order for each token]

        Args:
            clean_tokens (list(str)): [Token list in the clean text]
            ori_label (int): [The original label of the clean text]

        Returns:
            replace_list (list(original_index, new_token)): [Replace_list, a 2D array where each item includes the origianl_index and new_token]
            query_number (int): [query number for this function]
        """
        all_diff = []
        all_candidates = []
        query_number = 0
        all_saliency, saliency_list, new_query = self.position_saliency_decision(clean_tokens, ori_label)
        query_number += new_query
        result_list = []
        for i, w in enumerate(clean_tokens):
            # Replace the adj, adv, verb, noun only
            is_perturbed = False
            pos_ls = util.get_pos([w])
            pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
            for pos in pos_pref:
                if pos_ls[0] == pos:
                    is_perturbed = True

            # is_perturbed = True
            if not is_perturbed:
                all_diff.append([])
                all_candidates.append([w])
                continue

            if w in self.synonym_dict.keys():
                candidates = self.synonym_dict[w]
            else:
                candidates = self.replace_with_synonym_function(w)
                self.synonym_dict[w] = candidates
            all_candidates.append(candidates)
            max_word, max_diff, new_query, diff = self.candidate_word_saliency_decision(clean_tokens, i, candidates, ori_label)
            query_number += new_query
            result_list.append([i, max_word, max_diff])
            all_diff.append([float(s) for s in diff])

        

        # score_list = []
        # score_index_list = []
        # for i, max_word, max_diff in result_list:
        #     score_list.append(max_diff * saliency_list[i])
        #     score_index_list.append(i)
        score_list = [res[2] * saliency for res, saliency in zip(result_list, saliency_list)]
        score_index_list = [res[0] for res in result_list]
        indexes = np.argsort(np.array(score_list))[::-1]
        replace_list = []
        for index in indexes:
            res = result_list[index]
            replace_list.append([res[0], res[1]])

        return replace_list, query_number


    def position_saliency_decision(self, clean_tokens, ori_label):
        """[Compute the saliency of each word in the original word to decision the position importance]

        Args:
            clean_tokens (list(str)): [Token list in the clean text]
            ori_label (int): [The original label of the clean text]

        Returns:
            saliency (np.array(float), 1D): [Inlcuding the saliency of each word]
            soft_saliency_list (list(float, 1D)): [The softmax version of saliency]
            query_number (int): [query number for this function]
        """
        query_number = 0
        saliency_list = []
        probs, label = self.query_model(clean_tokens)
        score = probs[ori_label]
        query_number += 1

        all_sentence_new_list = []
        for i in range(len(clean_tokens)):
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[i] = '[UNK]'
            all_sentence_new_list.append(clean_tokens_new.copy())

        logits_new = self.query_model_group(all_sentence_new_list).numpy()
        score_new = self._softmax(logits_new)[:, ori_label]
        saliency = np.array(score - score_new)
        soft_saliency_list = list(self._softmax(saliency))
        query_number += len(logits_new)

        return saliency, soft_saliency_list, query_number
    

    def candidate_word_saliency_decision(self, clean_tokens, idx, candidates, ori_label):
        """[Compute the saliency of each candidate word to decision the word importance]

        Args:
            clean_tokens (list(str)): [Token list in the clean text]
            idx (int): [The idx-the word is operated]
            candidates (list(str)): [The candidate list of the idx-th word in the original text]
            ori_label ([type]): [The original label of the clean text]

        Returns:
            [type]: [description]
        """
        query_number = 0
        max_diff = -100
        max_word = clean_tokens[idx]
        probs, label = self.query_model(clean_tokens)
        score = probs[ori_label]
        query_number += 1

        sentence_new_list = []
        for c in candidates:
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[idx] = c
            sentence_new_list.append(clean_tokens_new.copy())
        if len(sentence_new_list) != 0:
            logits_new = self.query_model_group(sentence_new_list).numpy()
            query_number += len(logits_new)
            score_new = self._softmax(logits_new)[:, ori_label]
            diff = np.array(score - score_new)
            max_diff = np.max(diff)
            max_word = candidates[np.argmax(diff)]
        return max_word, max_diff, query_number, diff


    def replace_with_synonym_function(self, word):
        # candidate_word_list,_ = util.replace_with_synonym(word, 'nltk')
        candidate_word_list,_ = util.replace_with_synonym(word, 'embedding', idx2word= self.idx2word, word2idx=self.word2idx, cos_sim= self.cos_sim)
        candidate_word_list = candidate_word_list[:self.synonym_num+1]
        return candidate_word_list

    def query_model(self, text):
        probs = self.predictor([text])[0].data.cpu()
        label = torch.argmax(probs, dim=-1).data.numpy()
        return probs, label

    def query_model_group(self, text_list):
        probs = self.predictor(text_list).data.cpu()
        return probs

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x
    
    def check_diff(self, sentence, perturbed_sentence):
        words = sentence
        perturbed_words = perturbed_sentence
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count

    

    