import os
import time
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import SelectionDataset
from model import Model
import pickle


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_running_model(dataloader, test_mode):
    model.eval()
    pkl_name = '{}_{}_{}_{}_cache.pkl'.format(test_mode, args.test_data_dir, args.max_num_test_contexts, args.max_contexts_length)

    pkl_name = pkl_name.replace('/', '')
    if not os.path.exists(pkl_name):
        input_masks = []
        input_types = []
        str_keys = []
        print('pre-caching...')
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch[0].numpy()
            str_keys += [" ".join(item) for item in input_ids.astype(str)]
            input_masks += batch[1].numpy().tolist()
            input_types += batch[2].numpy().tolist()
        mapping = {}
        for str_key, masks, types in zip(str_keys, input_masks, input_types):
            if str_key not in mapping:
                mapping[str_key] = [str_key, masks, types]
        with open(pkl_name, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('loading...')
        with open(pkl_name, 'rb') as handle:
            mapping = pickle.load(handle)
   
    # pass mapping for encoder to encode
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
        print('encoding...')
        encoder_cache = model.encoder_inference(mapping)
        print('inferencing...')
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch[0].numpy()
            keys = [" ".join(item) for item in input_ids.astype(str)]
            struct_vec = [encoder_cache[key] for key in keys]
            struct_vec = torch.stack(struct_vec, 0)
            model.inference_forward(struct_vec.to(device), batch[3].to(device), args.max_num_test_contexts)
    
    tree_results = []
    relation_types = []
    for result in model.struct_attention.tree_results:
        tree_result, predicted_types = result
        tree_results += tree_result
        relation_types += predicted_types

    # clean up the tree results
    model.struct_attention.tree_results = []
    
    with open(os.path.join(args.test_data_dir, '{}_links.json'.format(test_mode))) as infile:
        gt = json.load(infile)
    res = []
    hits = 0
    cnt_preds = 0
    cnt_golds = 0

    for ds, g, r in zip(tree_results, gt, relation_types):
        all_d = set()
        if args.link_only:
            for d in ds:
                d = set([(dd, idx+1) for idx, dd in enumerate(d[1:])]) # skip -1
                all_d.update(d)
            g = set([tuple(gg[:2]) for gg in g])
        else:
            for d in ds:
                d = set([(dd, idx+1, r[dd][idx+1]) for idx, dd in enumerate(d[1:])]) # skip -1
                all_d.update(d)
            d = all_d
            g = set([tuple(gg) for gg in g])

        hits += len(d.intersection(g))
        cnt_golds += len(g)
        cnt_preds += len(d)
    prec = hits/cnt_preds
    rec = hits/cnt_golds
    f1 = 2*prec*rec/(prec+rec)

    return {'f1':f1}

def evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model):
    dev_result = eval_running_model(dev_dataloader, 'dev')
    test_result = eval_running_model(test_dataloader, 'test')
    print('Epoch %d, Global Step %d TST res:\n' % (epoch, global_step), dev_result)
    print('Epoch %d, Global Step %d TST res:\n' % (epoch, global_step), test_result)
    log_wf.write('Global Step %d VAL res:\n' % global_step)
    log_wf.write('Global Step %d TST res:\n' % global_step)
    log_wf.write(str(dev_result) + '\n')
    log_wf.write(str(test_result) + '\n')
    # save model
    if dev_result['f1'] > best_f1:
        # save model
        state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
        print('[Saving at]', state_save_path)
        log_wf.write('[Saving at] %s\n' % state_save_path)
        torch.save(model.state_dict(), state_save_path)

    return max(best_f1, dev_result['f1'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--encoder_model", required=True, type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_dir", default='/dev/null', type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--test_data_dir", type=str)

    parser.add_argument("--max_contexts_length", default=28, type=int, help="Number of tokens per context")
    parser.add_argument("--max_num_train_contexts", type=int, help="Number of train contexts")
    parser.add_argument("--max_num_dev_contexts", type=int, help="Number of dev contexts")
    parser.add_argument("--max_num_test_contexts", type=int, help="Number of test contexts")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")
    parser.add_argument("--link_only", action="store_true")
    parser.add_argument("--cross_domain", action="store_true")

    parser.add_argument("--use_scheduler", action="store_true", help='Whether to use scheduler for learning rate adjustment')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient Accumulation Step")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Warmup optimization steps percentage")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta_1", default=0.9, type=float, help="beta_1 for Adam optimizer")
    parser.add_argument("--beta_2", default=0.999, type=float, help="beta_2 for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=float('inf'), type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                            help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.test_data_dir is None:
        args.test_data_dir = args.data_dir
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    if not args.eval:
        train_dataset = SelectionDataset(os.path.join(args.data_dir, 'train.txt'), args, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=train_dataset.batchify_join_str, shuffle=True, num_workers=1)
        t_total = len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
        dev_dataset = SelectionDataset(os.path.join(args.test_data_dir, 'dev.txt'), args, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_dataset.batchify_join_str, shuffle=False, num_workers=1)
    test_dataset = SelectionDataset(os.path.join(args.test_data_dir, 'test.txt'), args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.batchify_join_str, shuffle=False, num_workers=1)

    encoder_config = AutoConfig.from_pretrained(os.path.join(args.encoder_model, 'config.json'))
    if not args.eval:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a')
        shutil.copy(os.path.join(args.encoder_model, 'config.json'), args.output_dir)
        shutil.copy(os.path.join(args.encoder_model, 'tokenizer.json'), args.output_dir)
        encoder = AutoModel.from_pretrained(args.encoder_model)
    else:
        encoder = AutoModel.from_config(encoder_config)

    model = Model(encoder_config, encoder=encoder, link_only=args.link_only).to(device)
    
    if args.eval:
        state_save_path = os.path.join(args.encoder_model, 'pytorch_model.bin')
        print('Loading parameters from', state_save_path)
        model.load_state_dict(torch.load(state_save_path, map_location=torch.device('cpu')))
        test_result = eval_running_model(test_dataloader, 'test')
        print(test_result)
        exit()
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta_1, args.beta_2), eps=args.adam_epsilon)
    if args.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total*args.warmup_ratio), num_training_steps=t_total
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0
    best_f1 = 0
    
    for epoch in range(1, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader)//args.gradient_accumulation_steps) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss = model(*batch, max_sent_len=args.max_num_train_contexts)
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    nb_tr_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    if args.use_scheduler:
                        scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                    global_step += 1

                    if nb_tr_steps and nb_tr_steps % args.print_freq == 0:
                        bar.update(min(args.print_freq, nb_tr_steps))
                        time.sleep(0.02)
                        print(global_step, tr_loss / nb_tr_steps)
                        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
                        if args.cross_domain:
                            best_f1 = evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model)
        
        best_f1 = evaluate(args, epoch, global_step, dev_dataloader, test_dataloader, best_f1, model)
