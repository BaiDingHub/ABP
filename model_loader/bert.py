# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification, BertConfig
from model_loader.bert_package.tokenization import BertTokenizer
from model_loader.bert_package.modeling import BertForSequenceClassification, BertConfig

class Bert(nn.Module):

    def __init__(self, pretrained_dir, nclasses, max_seq_length=128, batch_size=32, config = None):

        super(Bert, self).__init__()
        self.config = config
        self._gpu_init()
        
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)
        self._model_into_cuda(self.model)
        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length = max_seq_length, batch_size = batch_size)

        self.batch_size = batch_size
    
    def forward(self, input_ids, input_mask, segment_ids):
        logits = self.model(input_ids, segment_ids, input_mask)
        return logits

    def text_pred(self, text_data):

        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size = self.batch_size)

        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)
    
    def input_to_embs(self, text):
        dataloader = self.dataset.transform_text(text, batch_size = self.batch_size)
        embs = []
        input_ids_list = []
        probs_all = []
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            emb = self.model.input_to_embs(input_ids, segment_ids, input_mask)
            embs.append(emb)
            input_ids_list.append(input_ids)

        return torch.cat(input_ids_list, dim=0), torch.cat(embs, dim=0)
    
    def embs_to_logits(self, input_ids, embs):
        outs = []
        # embs = embs.to(self.device)
        logits = self.model.embs_to_logit(input_ids, embs)
        outs.append(nn.functional.softmax(logits, dim=-1))

        return torch.cat(outs, dim=0)
    
    def get_embeddings(self):
        embeddings = self.model.get_embeddings()
        return embeddings
    
    def get_tokenizer(self):
        return self.dataset.tokenizer

    def _gpu_init(self):
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

    def _model_into_cuda(self, model):
        if self.use_gpu:
            model = model.to(self.device)
            if len(self.device_ids) > 1:
                model = nn.DataParallel(model, device_ids = self.device_ids)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.
    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader

    

def load_Bert(pretrained_dir, nclasses, max_seq_length, batch_size, config):
    bert = Bert(pretrained_dir, nclasses, max_seq_length, batch_size, config)
    return bert

