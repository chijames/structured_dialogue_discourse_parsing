import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random
import pickle

class SelectionDataset(Dataset):
    def __init__(self, file_path, args, tokenizer, sample_cnt=None):
        self.max_contexts_length = args.max_contexts_length
        self.data_source = []
        self.tokenizer = tokenizer
        if 'train' in file_path:
            max_num_contexts = args.max_num_train_contexts
        elif 'dev' in file_path:
            max_num_contexts = args.max_num_dev_contexts
        else:
            max_num_contexts = args.max_num_test_contexts

        with open(file_path) as f:
            i = 0
            group = {
                'context': [],
                'labels': [],
                'length': -1
            }
            for line in f:
                line = line.strip()
                if (i+1)%(max_num_contexts+1) != 0: # text
                    group['context'].append(line)
                else:
                    parents, length, relation_types = line.split('|')
                    length = int(length.strip())
                    relation_types = [int(num) for num in relation_types.strip().split()]
                    parents = [int(num) for num in parents.strip().split()]
                    rows = []
                    cols = []
                    for child, parent in enumerate(parents):
                        if parent == -1:
                            continue

                        rows.append(parent)
                        cols.append(child)
                        if parent >= child:
                            print (' '.join(map(str, parents)))
                            print (parent, child)
                            print('error')
                            exit()
                    group['labels'] = [rows, cols, relation_types]
                    group['length'] = length
                    self.data_source.append(group)
                    group = {
                        'context': [],
                        'labels': [],
                        'length': -1
                    }
                i += 1
                if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                    break
                
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        return self.data_source[index]

    def batchify_join_str(self, batch):
        contexts_pair_ids_batch, contexts_pair_ids_mask_batch, type_ids_batch, contexts_length_mask_batch, labels_batch = [], [], [], [], []
        for sample in batch:
            contexts, labels, length = sample['context'], sample['labels'], sample['length']
            first = []
            second = []
            for i in range(len(contexts)):
                for j in range(i, len(contexts)):
                    if i == j:
                        first.append("this is the placeholder for start of dialogue") # can be any dummy sentence
                    else:
                        first.append(contexts[i])
                    second.append(contexts[j])
            
            tokenized_dict = self.tokenizer(first, second, padding='max_length', truncation='longest_first', max_length=self.max_contexts_length*2)
            input_ids, attention_mask, type_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['attention_mask']
            contexts_pair_ids_batch += input_ids
            contexts_pair_ids_mask_batch += attention_mask
            type_ids_batch += type_ids
            contexts_length_mask_batch.append(length)
            labels_batch.append(labels)
        
        type_ids_batch = torch.LongTensor(type_ids_batch)
        contexts_pair_ids_batch = torch.LongTensor(contexts_pair_ids_batch)
        contexts_pair_ids_mask_batch = torch.LongTensor(contexts_pair_ids_mask_batch)
        contexts_length_mask_batch = torch.LongTensor(contexts_length_mask_batch)
        labels_batch = torch.LongTensor(labels_batch)

        return contexts_pair_ids_batch, contexts_pair_ids_mask_batch, type_ids_batch, contexts_length_mask_batch, labels_batch
