# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from datetime import datetime
import tempfile
import socket
import random
from itertools import combinations 

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2LMHeadModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from datasets import load_dataset, Dataset
import pandas as pd

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

da_map = {0: 'Wh-Question', 1: 'Other Answers', 2: 'Declarative Wh-Question', 3: 'Other', 4: 'Tag-Question', 5: 'Agree/Accept', 6: 'Statement-non-opinion', 7: 'Open-Question', 8: 'Conventional-closing', 9: 'Declarative Yes-No-Question', 10: 'Repeat-phrase', 11: 'Maybe/Accept-part', 12: 'Statement-opinion', 13: 'Appreciation', 14: 'Uninterpretable', 15: 'Offers, Options Commits', 16: 'Apology', 17: 'Action-directive', 18: 'Downplayer', 19: 'Acknowledge (Backchannel)', 20: 'Signal-non-understanding', 21: 'Hold Before Answer/Agreement', 22: 'Thanking', 23: 'Quotation', 24: '3rd-party-talk', 25: 'Response Acknowledgement', 26: 'Self-talk', 27: 'Rhetorical-Question', 28: 'Yes-No-Question', 29: 'Summarize/Reformulate', 30: 'Dispreferred Answers', 31: 'Backchannel in Question Form', 32: 'Negative Non-no Answers', 33: 'Affirmative Non-yes Answers', 34: 'Reject', 35: 'Collaborative Completion', 36: 'Yes Answers', 37: 'Hedge', 38: 'No Answers', 39: 'Conventional-opening', 40: 'Or-Clause'}
da_list = list(da_map.values())

# SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", '<bof>', "<pad>", "<bor>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>', '<boc>', '<boa>', '<eoa>']+da_list}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]

encoder_length = 512
decoder_length = 128
batch_size = 1
eval_batch_size = 1
column_to_remove = ['Unnamed: 0', 'Unnamed: 0.1', 'context', 'context_last', 'prediction', 'act_obj', 'rake_phrases', 'data_id', 'dialogue_acts', 'did', 'strategies', 'template', 'uit', 'utterance']

logger = logging.getLogger(__file__)

use_agenda_text = True


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

def build_input_from_segments(history, agenda_text, reply, lm_labels=True, max_history=4,\
                              with_eos=True, is_test=False, agenda_done=True, tokenizer=None):
    history = history[-max_history:]
    # if len(' '.join(history))>2900: history = history[-2:]
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    boc, eos, speaker1, speaker2, pad, bok, eok = '<boc>', '<eos>', '<speaker1>', '<speaker2>', '<eos>', '<boa>', '<eoa>' 

    # if agenda_text != '': agenda_text = '<bok> ' + agenda_text + ' '
    if agenda_done:
        sequence = [ boc +  ' '] + [h for h in history] + [bok + ' ' + agenda_text + ' ' + eok] +  [reply + str(' '+ eos if with_eos else '')]
        sequence = [sequence[0]] + [speaker2 + ' ' + s if (len(sequence) - i) % 2 else speaker1 + ' ' + s for i, s in enumerate(sequence[1:-1])] + [ sequence[-1]] # -1 to remove the current response
    else:
        sequence = [ boc +  ' '] + [h for h in history] + [bok + ' ' + agenda_text ]
        sequence = [sequence[0]] + [speaker2 + ' ' + s if (len(sequence) - i +1) % 2 else speaker1 + ' ' + s for i, s in enumerate(sequence[1:])]            

    # import pdb;pdb.set_trace()
    # print(sequence)

    # sequence = tokenize(sequence)
    # bos, eos, speaker1, speaker2, bof, pad, bor = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    instance = {}
    instance["sequence"] = sequence#list(chain(*sequence))
    # sequence = tokenizer(instance["sequence"], padding="max_length", truncation=True, max_length=encoder_length)
    sequence = tokenize(sequence)
    instance["input_ids"] = list(chain(*sequence))
    # sequence = tokenize(sequence)
    if agenda_done:
        instance["token_type_ids"] = [0 if (len(sequence) - i+1) % 2 else 1 for i, s in enumerate(sequence[:-1]) for _ in s] + [0 for i, s in enumerate(sequence[-1])]
    else:
        instance["token_type_ids"] = [0 if (len(sequence) - i) % 2 else 1 for i, s in enumerate(sequence) for _ in s]
    # instance["sequence"] = ' '.join(instance["sequence"]).replace('  ', ' ')
    # global max_length
    # max_length = max(max_length, len(instance["sequence"]))
    # instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    # sequence = inputs.input_ids
    # import pdb;pdb.set_trace()
    instance["labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["labels"] = ([-100] * sum(len(s) for s in sequence[:-2])) +  sequence[-2] + sequence[-1]
    if is_test:
        instance["labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) +  sequence[-1]
    # import pdb;pdb.set_trace()
    return instance

def get_context_text(context_list):
    context_list = eval(context_list)
    history = []
    for i, h in enumerate(context_list):
        if type(h['text']) == dict and "price" in h['text']:
            h['text'] = 'price offerred is ' + str(h['text']['price'])
        if h['text'] == None: h['text'] = 'none'
        history.append(str(h['text']))

    return history

def get_agenda_batch(batch, agenda_prob = 1.0):

    if type(batch['context'])==list:
        batch['context'] = batch['context'][0]
    responses = []
    agenda_choices = []
    # import pdb;pdb.set_trace()
    for i,_ in enumerate(batch['response']):
        da = batch['prediction'][i]
        act_obj = ""#batch['act_obj'][i]
        rake_phrases = batch['rake_phrases'][i]
        response = batch['response'][i]
        responses.append(response)

        #set phrases_to_use to act_obj list, if not then rake_phrases list, if not then empty list
        phrases_to_use = act_obj
        if phrases_to_use=='' or phrases_to_use==None:
            phrases_to_use = rake_phrases
        if phrases_to_use is not None:
            phrases_to_use = phrases_to_use.split('; ')
        else:
            phrases_to_use = []
        if len(phrases_to_use)!=0:
            for phrase in phrases_to_use:
                agenda_choices.append(da + ' ' + phrase)

    phrases_to_use = []
    use_agenda = True if random.random()<agenda_prob else False
    if len(agenda_choices)==0: use_agenda = False 
    if not use_agenda: agenda_choices = ['']
    agenda_text = random.choice(agenda_choices)

    batch['response'] = ' '.join(responses)
    batch['agenda_text'] = agenda_text

    return batch


def get_all_agendas_batch(batch, agenda_prob = 1.0):

    if type(batch['context'])==list:
        batch['context'] = batch['context'][0]
    responses = []
    agenda_choices = []
    # import pdb;pdb.set_trace()
    for i,_ in enumerate(batch['response']):
        da = batch['prediction'][i]
        act_obj = ""#batch['act_obj'][i]
        rake_phrases = batch['rake_phrases'][i]
        response = batch['response'][i]
        responses.append(response)

        #set phrases_to_use to act_obj list, if not then rake_phrases list, if not then empty list
        phrases_to_use = act_obj
        if phrases_to_use=='' or phrases_to_use==None:
            phrases_to_use = rake_phrases
        if phrases_to_use is not None:
            phrases_to_use = phrases_to_use.split('; ')
        else:
            phrases_to_use = []
        if len(phrases_to_use)!=0:
            for phrase in phrases_to_use:
                # agenda_choices.append(da + ' ' + phrase)
                agenda_choices.append(phrase)

    # use_agenda = True if random.random()<agenda_prob else False
    # if len(agenda_choices)==0: use_agenda = False 
    # if not use_agenda: agenda_choices = ['']
    # agenda_text = random.choice(agenda_choices)

    batch['response'] = ' '.join(responses)
    agenda_list = []
    if len(agenda_choices)>5:
        agenda_choices = random.sample(agenda_choices,5)
    for r in range(len(agenda_choices)):
        r_list = list(combinations(agenda_choices, r)) 
        agenda_list.append(random.choice(r_list))
    
    # batch['agenda_texts'] = ['']
    batch['agenda_texts'] = []
    for agenda_c in agenda_list:
        batch['agenda_texts']+=[' <sep> '.join(agenda_c)]
    # batch['agenda_texts'] += [' <sep> '.join(agenda_choices)]
    if len(batch['agenda_texts'])==0: batch['agenda_texts'] = ['']
    return batch


def get_full_agendas_batch(batch, agenda_prob = 1.0):

    if type(batch['context'])==list:
        batch['context'] = batch['context'][0]
    responses = []
    agenda_choices = []
    # import pdb;pdb.set_trace()
    for i,_ in enumerate(batch['response']):
        rake_phrases = batch['rake_phrases'][i]
        response = batch['response'][i]
        responses.append(response)

        phrases_to_use = rake_phrases
        if phrases_to_use is not None:
            phrases_to_use = phrases_to_use.split('; ')
        else:
            phrases_to_use = []
        if len(phrases_to_use)!=0:
            for phrase in phrases_to_use:
                # agenda_choices.append(da + ' ' + phrase)
                agenda_choices.append(phrase)



    # agenda_choices = [' '.join(agenda_choices)]
    batch['response'] = ' '.join(responses)

    if len(agenda_choices)>10:
        agenda_choices = agenda_choices[:10]
    # batch['agenda_texts'] = ['']
    batch['agenda_texts'] = []
    batch['agenda_texts']+=[' <sep> '.join(agenda_choices)]
    # batch['agenda_texts'] += [' <sep> '.join(agenda_choices)]
    if len(batch['agenda_texts'])==0: batch['agenda_texts'] = ['']
    return batch

# map data correctly
def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
    # use bert tokenizer here for encoder

    items = []
    keys = list(batch.keys())
    # print(examples['context'])
    for i in range(len(batch[keys[0]])):
        ex = {}
        for k in keys:
            ex[k] = batch[k][i]
        items.append(ex)

    result = defaultdict(list)
    for b,dp in enumerate(items): #loop over all data points in batch
        # print(dp)
        dp = get_full_agendas_batch(dp)

        response = dp["response"]
        # if response == None: response = 'none'
        response = str(response)

        context_list = dp["context"]
        history = dp["context"].split(' _eos')

        if not use_agenda_text:
            agenda_text = ''
            dp['agenda_texts'] = ['']

        # print(history, response, dp['agenda_texts'])
        # import pdb;pdb.set_trace()
        agenda_choices = dp['agenda_texts']
        agenda_choices = list(set(agenda_choices))
        # import pdb;pdb.set_trace()
        is_test = False
        if 'is_test' in dp:
            is_test = True
            #during validation we will take the whole agenda as input, which is the longest string in all agendas
            agenda_choices = [max(agenda_choices, key = len)]

        for agenda_text in agenda_choices:
            instance = build_input_from_segments(history, agenda_text, response, is_test=is_test, tokenizer=tokenizer)
            dp['input_ids'] = instance['input_ids']
            dp["labels"] = instance["labels"]
            dp["token_type_ids"] = instance["token_type_ids"]

            for key, value in dp.items():
                result[key].append(value)


    for colrname in column_to_remove:
        if colrname in batch:
            del batch[colrname]

    return result



def collate_fn(examples):
    print(examples)
    return tokenizer.pad(examples, return_tensors='pt', padding=True)

def collate_fn(examples):
    # print(examples)

    batch_dict = defaultdict(list)
    # import pdb;pdb.set_trace()

    for input_name in examples[0].keys():
        for e in examples:
            batch_dict[input_name].append(e[input_name])
    tensors = pad_and_tensorize(batch_dict, padding=tokenizer.pad_token_id)
    return tensors

def pad_and_tensorize(batch_dict, padding):
    #https://github.com/sshleifer/transfer-learning-conv-ai/blob/batch-padding/train.py
    """ Pad the batch_dict."""
    tensors = []
    # import pdb;pdb.set_trace()

    for name in batch_dict.keys():
        if name not in PADDED_INPUTS:
            tensors.append(torch.tensor(batch_dict[name]))
            continue
        entry = batch_dict[name]
        pad_id = padding if name != "labels" else -100
        # padded = pad_sequence([torch.tensor(seq) for x in entry for seq in x], batch_first=True, padding_value=pad_id)
        padded = pad_sequence([torch.tensor(x) for x in entry ], batch_first=True, padding_value=pad_id)

        # bs, n_candidates = len(entry), len(entry[0])
        # tensors.append(padded.view(bs, n_candidates, -1))
        bs = len(entry)
        tensors.append(padded.view(bs, -1))

    return tensors


def aggregate_rows(df):
    result_df = df.groupby('data_id').agg(lambda x: list(x))
    del df

    return result_df

def get_data_loaders(args, tokenizer):

    train_file = args.train_file #"data/train_da_key.csv"
    validation_file = args.validation_file #"data/valid_da_key.csv"
    ##for loading from csv into pandas and aggregating first 
    train_df = pd.read_csv(train_file)#[:1000]
    valid_df = pd.read_csv(validation_file)#[:1000]
    train_df = aggregate_rows(train_df)
    valid_df = aggregate_rows(valid_df)
    valid_df['is_test'] = 'True'

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    train_dataset = train_dataset.map(
        map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size#, remove_columns=column_to_remove,
    )
    train_dataset.set_format(
        # type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        type="torch", columns=["input_ids","labels", "token_type_ids"],

    )
    print(len(train_dataset))

    valid_dataset = valid_dataset.map(
        map_to_encoder_decoder_inputs, batched=True, batch_size=eval_batch_size#, remove_columns=column_to_remove,
    )

    valid_dataset.set_format(
        # type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        type="torch", columns=["input_ids","labels", "token_type_ids"],
    )
    # va


    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed),
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False,
                              collate_fn=collate_fn)
    return train_loader, valid_loader, train_sampler, valid_sampler

def train():
    parser = ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--train_file", type=str, default="data/sample100.csv", help="path to train csv")
    parser.add_argument("--validation_file", type=str, default="data/sample100.csv", help="path to test csv")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer #if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    global tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2LMHeadModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        # print(batch)
        # import  pdb;pdb.set_trace()
        # exit(0)
        # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        # (lm_loss), (mc_loss), *_ = model(
        #     input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        #     mc_labels=mc_labels, lm_labels=lm_labels
        # )
        # loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        input_ids, lm_labels, token_type_ids = batch

        (lm_loss), *_ = model(input_ids, token_type_ids=token_type_ids, labels=lm_labels)
        # lm_loss = model(input_ids, token_type_ids=token_type_ids, labels=lm_labels)['loss']
        loss = (lm_loss * args.lm_coef) / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            # input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            # lm_logits, mc_logits, *_ = model(
            #     input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            # )
            input_ids, lm_labels, token_type_ids = batch
            lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
            # lm_logits = model(input_ids, token_type_ids=token_type_ids, labels=lm_labels)['logits']

            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted), (lm_labels_flat_shifted)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    # metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
    #            "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    # metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
    #                 "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=None)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()

