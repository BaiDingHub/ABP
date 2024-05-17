import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import data_loader
import utils
import model_loader
import config as config_module
from tqdm import tqdm
from torch.utils.data import DataLoader, BatchSampler
from transformers import AlbertTokenizer, BertTokenizer

def train_model(epoch, model, optimizer,
        train_x, train_y,
        test_x, test_y,
        best_test, save_path, device):

    model.train()
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for step, (x, y) in enumerate(tqdm(zip(train_x, train_y), desc="Iteration")):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x).to(device), Variable(y).to(device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    test_acc = eval_model(niter, model, test_x, test_y, device)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.item(),
        test_acc
    ))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
        # test_err = eval_model(niter, model, test_x, test_y)
    sys.stdout.write("\n")
    return best_test

def eval_model(niter, model, input_x, input_y, device):
    model.eval()
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.
    # total_loss = 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(zip(input_x, input_y), desc="Iteration")):
            x, y = Variable(x, volatile=True).to(device), Variable(y).to(device)
            output = model(x)
            # loss = criterion(output, y)
            # total_loss += loss.item()*x.size(1)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum()
            cnt += y.numel()
    model.train()
    return correct.item()/cnt

def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type+'.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))

def get_gpu_device(config):
    use_gpu = False
    device_ids = [0]
    device = torch.device('cpu')
    if config.GPU['use_gpu']:
        if not torch.cuda.is_available():
            print("There's no GPU is available , Now Automatically converted to CPU device")
        else:
            message = "There's no GPU is available"
            device_ids = config.GPU['device_id']
            assert len(device_ids) > 0,message
            device = torch.device('cuda', device_ids[0])
            use_gpu = True
    
    return device


def train_bert_model(epoch, model, optimizer,
        train_loader, test_loader, best_test, save_path, device):

    model.train()
    niter = epoch*len(train_loader)
    loss_list = []
    correct = 0.0
    cnt = 0
    criterion = nn.CrossEntropyLoss()
    # import pdb; pdb.set_trace()
    for step, (input_ids, input_mask, segment_ids, label) in enumerate(tqdm(train_loader, desc="Iteration")):
        niter += 1
        cnt += 1
        model.zero_grad()
        input_ids, input_mask, segment_ids, label = Variable(input_ids.to(device)), Variable(input_mask.to(device)), Variable(segment_ids.to(device)), Variable(label.to(device))
        output = model(input_ids, input_mask, segment_ids)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
        cnt += label.numel()
        
        label = label.cpu().data
        pred = pred.cpu().data
    
    train_acc = correct/cnt
    print("Epoch={} Loss={} train_acc={:.6f}".format(epoch, np.mean(loss_list), train_acc))
    test_acc = eval_bert_model(niter, model, test_loader, device)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} train_acc = {:.6f} test_acc={:.6f}\n".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.item(), train_acc,
        test_acc
    ))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
    sys.stdout.write("\n")
    return best_test

def eval_bert_model(niter, model, test_loader, device):
    model.eval()
    correct = 0.0
    cnt = 0.
    with torch.no_grad():
        for step, (input_ids, input_mask, segment_ids, label) in enumerate(tqdm(test_loader, desc="Iteration")):
            input_ids, input_mask, segment_ids, label = Variable(input_ids.to(device)), Variable(input_mask.to(device)), Variable(segment_ids.to(device)), Variable(label.to(device))
            output = model(input_ids, input_mask, segment_ids)
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data).cpu().sum()
            cnt += label.numel()
            
            label = label.cpu().data
            pred = pred.cpu().data
    correct= correct.item()

    model.train()
    return correct/cnt

def main(args):
    config = config_module.Config()

    device = torch.device('cuda', args.gpu_id)

    train_x, train_y = data_loader.read_train_text(args.dataset, shuffle=True)
    test_x, test_y = data_loader.read_corpus(args.test_path, shuffle=True)

    nclasses = max(train_y) + 1

    # if args.model == 'cnn':
    #     model = model_loader.WordCNN(embedding = args.embedding_path, hidden_size=150, dropout=0.3, batch_size = args.batch_size, nclasses= nclasses, max_seq_length = args.max_seq_length, config=config)
    #     device = get_gpu_device(config)
    #     model.to(device)
    if args.model == 'lstm':
        model = model_loader.WordLSTM(embedding = args.embedding_path, hidden_size=150, depth=1, dropout=0.3, batch_size = args.batch_size, nclasses= nclasses, max_seq_length = args.max_seq_length, config = config)
        device = get_gpu_device(config)
        model.to(device)
    elif args.model == 'albert':
        model = model_loader.AlBert(args.pretrained_dir, args.nclasses, args.max_seq_length, args.batch_size, config)
        tokenizer = AlbertTokenizer.from_pretrained(args.pretrained_dir, do_lower_case=True)
        model.to(device)

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    if 'bert' in args.model:
        train_dataset = data_loader.MyTrainDataset(dataset = args.dataset, data_dir = './data/dataset', mode='train', is_bert=True, max_seq_length=args.max_seq_length, tokenizer = tokenizer, model_name=args.model)
        test_dataset = data_loader.MyTrainDataset(dataset = args.dataset, data_dir = './data/dataset', mode='test', is_bert=True, max_seq_length=args.max_seq_length, tokenizer = tokenizer, model_name=args.model)

        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
    else:
        train_x, train_y = utils.create_batches(
            train_x, train_y,
            args.batch_size,
            model.word2id,
            args.max_seq_length
        )
        test_x, test_y = utils.create_batches(
            test_x, test_y,
            args.batch_size,
            model.word2id,
            args.max_seq_length
        )

    best_test = 0
    if 'bert' in args.model:
        for epoch in range(args.max_epoch):
            best_test = train_bert_model(epoch, model, optimizer,
                train_loader,
                test_loader,
                best_test, args.save_path, device
            )
            if args.lr_decay>0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay
    else:
        for epoch in range(args.max_epoch):
            best_test = train_model(epoch, model, optimizer,
                train_x, train_y,
                test_x, test_y,
                best_test, args.save_path, device
            )
            if args.lr_decay>0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay


    sys.stdout.write("test_err: {:.6f}\n".format(
        best_test
    ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--model", type=str, default="cnn", help="which model")
    argparser.add_argument("--dataset", type=str, default="sst", help="which dataset")
    argparser.add_argument("--test_path", type=str, default="./data/sst/sst", help="test dataset")
    argparser.add_argument("--embedding_path", type=str, default="./data/embedding/glove.6B.200d.txt", help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--nclasses", type=int, default=2, help='The number of class')
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='./checkpoint')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--max_seq_length", type=int, default=256)
    argparser.add_argument("--pretrained_dir", type=str, default='')
    

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print (args)
    # torch.cuda.set_device(args.gpu_id)
    main(args)
    
