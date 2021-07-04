# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from pprint import pformat
import warnings
import os
import csv
import json
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from itertools import combinations 

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPT2DoubleHeadsModel
# from utils import get_dataset, download_pretrained_model
from rake_nltk import Rake
logger = logging.getLogger(__file__)

from datasets import load_dataset, Dataset
import pandas as pd
import re
use_agenda_text = True
type_experiment = 'gpt_multimap_al'
from train_agenda_gpt_multimapal import build_input_from_segments, add_special_tokens_, get_agenda_batch
from torch.multiprocessing import Pool, Process, set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
torch.set_num_threads(10)
DATA_LIMIT=None
DATA_START=None

SPECIAL_TOKENS = ['<boc>', '<eos>', '<speaker1>', '<speaker2>', '<eos>', '<boa>', '<eoa>']

p = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+))')

import gensim.downloader as api

# Download the models
# fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
# word2vec_model300 = api.load('word2vec-google-news-300')
# print(fasttext_model300.most_similar(positive='king', negative=None, topn=5, restrict_vocab=None, indexer=None))
# print(glove_model300.most_similar_cosmul(positive='king', negative=None, topn=5))

def cap(match):
    return(match.group().capitalize())

def clean_artifacts(response):
    clean_response = response.strip()
    sentence_words = clean_response.split()
    final_response = []
    for i, word in enumerate(sentence_words):
        if i>1 and word==sentence_words[i-1]:
            continue
        final_response.append(word)
    clean_response = ' '.join(final_response)
    clean_response = clean_response.replace(' ,', ',')
    clean_response = clean_response.replace(' .', '.')
    clean_response = clean_response.replace(' ?', '?')
    clean_response = clean_response.replace('  ', ' ')
    clean_response = clean_response.replace(' !', '!')
    clean_response = clean_response.replace(' \' ', '\'')
    clean_response = clean_response.replace(' \'', '\'')
    clean_response= re.sub(' +',' ',clean_response)
    clean_response = re.sub('\$ ', '$', re.sub(' \%', '%', clean_response))
    clean_response = re.sub('\. (\d+)', r'.\1', clean_response)
    clean_response = p.sub(cap, clean_response)
#     clean_response = re.sub(r'($)?\s+(\d)(.)?\s+(\d)', r'\1\2', clean_response)

    return clean_response

def read_json_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
            
    return data

    
def save_data_to_disk(data, dest_file):
    output_file = open(dest_file, 'w', encoding='utf-8')
    for i,dic in enumerate(data):
        # if DATA_LIMIT and i==DATA_LIMIT:break
        json.dump(dic, output_file)
        if i!=len(data): 
            output_file.write("\n")

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def sample_sequence(history, agenda_text, tokenizer, model, args, current_output=None):
    special_tokens_ids = []#tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        # import pdb;pdb.set_trace()
        instance = build_input_from_segments(history, agenda_text, tokenizer.decode(current_output), tokenizer=tokenizer, with_eos=False, max_history=3)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        # logits = model(input_ids, token_type_ids=token_type_ids)
        logits = model(input_ids, token_type_ids=token_type_ids)['logits']

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            tries = 0
            while prev.item() in special_tokens_ids and tries< args.max_length:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)
                tries+=1
        if prev.item() == tokenizer.eos_token_id:#in special_tokens_ids:
            break
        current_output.append(prev.item())

    return tokenizer.decode(current_output, skip_special_tokens=True)

def aggregate_rows(df):
    result_df = df.groupby('data_id').agg(lambda x: list(x))
    del df

    return result_df

poor_neighbors = ["nothing", "really", "please", "nothing", "anything", "if", "n't", "these", "would", "will", "those", "this", "all", "to", "each", "we", "be", "any", "so"]


def get_sim_keywords(keyphrases, glove_model300):
    keyphrases_neighbours = dict()
    for keyphrase in keyphrases:
        keywords = keyphrase.split()
        keyphrase_list = []
        for keyword in keywords:
            try:
                sim_word_list = glove_model300.most_similar_cosmul(positive=keyword.lower(), negative=None, topn=10)
                # to_add = random.sample(sim_word_list,1)
                keyphrase_list+=sim_word_list
            except:
                pass
        calculated = [x[0] for x in keyphrase_list]
        calculated = [x for x in calculated if x not in poor_neighbors]
        keyphrases_neighbours[keyphrase] = calculated + [keyphrase]

    return keyphrases_neighbours




def get_model_outputs(data_point, data_points, tokenizer, model, glove_model300, args, desired_number_responses=10):
    rake = Rake()
        # tensor_text, input_text = (batch[0]), batch[1]
        # history, utterance, tokenizer = data_point
    random_dp = random.choice(data_points)
    history = random_dp['context']
    context_list = data_point['context']
    context_keywords = rake.extract_keywords_from_text(' '.join(context_list))
    keyphrases = rake.get_ranked_phrases()
    # print('---', context_list)
    # print('---random context: ', history)
    # print(keyphrases)
    # keyphrases = [' '.join(w.split()[:2]) for w in keyphrases]
    agenda_list = []
    # if len(keyphrases)>3:
    #     keyphrases = random.sample(keyphrases,3)
    if len(keyphrases) ==0: keyphrases = ['']
    keyphrases = [x for x in keyphrases if len(x)>2]
    if len(keyphrases)>20:
        keyphrases = random.sample(keyphrases, 20)

    keyphrases_neighbours = get_sim_keywords(keyphrases, glove_model300)
    # print(keyphrases_neighbours)
    while len(agenda_list)<2*desired_number_responses:
        for r in range(2, 4):
            r_list = list(combinations(keyphrases, r))
            if len(r_list)==0:
                 r_list = ['']
            # agenda_list.append(random.choice(r_list))
            # to add similar word instead
            selected_list = random.choice(r_list)
            # print('selected', selected_list)
            new_rlist = []
            for phrase in selected_list:
                if random.random()<0.7:
                    new_rlist.append(phrase)
                else:
                    neighbourlist = keyphrases_neighbours.get(phrase)
                    chosen_neighbour = random.sample(neighbourlist,1)[0]
                    new_rlist.append(chosen_neighbour)

            # print('new_rlist', new_rlist)
            agenda_list.append(new_rlist)
            if len(agenda_list)>=2*desired_number_responses:
                break

    agenda_texts = []
    for agenda_c in agenda_list:
        ta = ' <sep> '.join(agenda_c)
        if len(ta.split())<=6:
            agenda_texts+=[ta]
    # print('agenda_texts: ', agenda_texts)
    outs = []
    # for agenda_text in agenda_texts:
    it = 0
    while len(outs)<desired_number_responses:
        # instance = build_input_from_segments(history, agenda_text, '', tokenizer=tokenizer)
        # print("agenda_text --> ", agenda_text)
        it+=1
        random_dp = random.choice(data_points)
        history = random_dp['context']

        # history_new = []
        # total_length = 0
        # for h in history[::-1]:
        #     total_length+=len(h.split())
        #     if total_length>350:
        #         break
        #     else:
        #         history_new.insert(0, h)

        # history = history_new


        agenda_text = agenda_texts[it%len(agenda_texts)]

        # print('--', history)
        # print(agenda_text)
        text_generated = sample_sequence(history,
                                           agenda_text,
                                           tokenizer,
                                           model,
                                           args,
                                           current_output=None
                                           )
        
        if text_generated!='' and text_generated not in outs:# and len(text_generated.split())>8:
            text_generated = clean_artifacts(text_generated)
            # print(agenda_text, ' - ', text_generated)
            outs.append(text_generated)

    return outs

def map_to_gen(i, data_point, data_points, tokenizer, model, glove_model300, args, desired_number_responses):
    # print(i)
    # print(data_point)
    text_list = get_model_outputs(data_point, data_points, tokenizer, model, glove_model300, args, desired_number_responses=desired_number_responses)
    # print(i,text_list)
    if i%100==0:
        print(i)


    return i, text_list

def evaluate(args, model, glove_model300, tokenizer, dataset, prefix="", frames_type ="response_frames", desired_number_responses=20):
    eval_output_dir = args.output_dir

    # dat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    logger.info("Build inputs and labels")
    data_points = dataset#make_data_lists(args, dat, tokenizer)
    # pad_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    print('evaluating: ', len(dataset))
    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    model.share_memory()
    # Eval!
    prefix += '_'+str(DATA_START) + '-' + str(DATA_LIMIT)
    output_eval_file = eval_output_dir + prefix + "_kw.json"
    print("Saving at", output_eval_file)
    logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.valid_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    if not os.path.exists(os.path.join(eval_output_dir, prefix)):
        os.makedirs(os.path.join(eval_output_dir, prefix))
    

    data_to_process = []
    for i, data_point in enumerate(data_points):
        if DATA_START and i<DATA_START: continue
        if DATA_LIMIT and i==DATA_LIMIT:break
      # history, utterance, frames, tokenizer = data_point
      # print([outs[i]])
        # if i<6549 or i>6555: continue
        data_to_process.append([i, data_point, data_points, tokenizer, model, glove_model300, args, desired_number_responses])
    pool = Pool(3)
    result_objects = [pool.apply_async(map_to_gen, args=(a)) for a in data_to_process]

    results = []
    for job in tqdm(result_objects):
        results.append(job.get())
    pool.close()
    pool.join()
    # results = [r.get() for r in result_objects]
    # result_objects = [(map_to_sample(*a)) for a in to_process]
    # print(results)
    results.sort(key=lambda x: x[0])

    data_to_process = [x[1] for x in data_to_process]
    for i, (result, data_point) in enumerate(zip(results, data_to_process)):
        ind, text_list = result
        data_point['adv_gen_neg_responses_t1'] = text_list
      ## print(ind, data_point)
      ## print(result)

    # data_to_process = []
    # for i, data_point in tqdm(enumerate(data_points)):
    #     if DATA_START and i<DATA_START: continue
    #     if DATA_LIMIT and i==DATA_LIMIT:break
    #     outs = get_model_outputs(data_point, data_points, tokenizer, model, glove_model300, args, desired_number_responses=desired_number_responses)
    #     data_point['adv_gen_neg_responses_t1'] = outs
    #     data_to_process.append(data_point)

    # if not os.path.exists(os.path.join(eval_output_dir, prefix)):
    #     os.makedirs(os.path.join(eval_output_dir, prefix))

    save_data_to_disk(data_to_process, output_eval_file)
    # pool.close()

    return True


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)",
                        choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=40, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.9, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--type_experiment", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--given_da", type=str, default=None, help="")
    parser.add_argument("--eval_input_outputs", type=str, default="eval_input_outputs.csv", help="")
    parser.add_argument("--eval_outputs", type=str, default="eval_outputs.txt", help="")
    parser.add_argument("--simple_da_adjustment_weight", type=float, default=0.1, help="")
    parser.add_argument('--data_limit', type=int)
    parser.add_argument('--data_start', type=int)
    parser.add_argument("--input_file", type=str, default="data/train_keyword_input.json", help="input json file")

    args = parser.parse_args()
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # args.distributed = (args.local_rank != -1)
    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     args.device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.output_dir = args.model_checkpoint
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    global DATA_LIMIT
    global DATA_START
    if args.data_start:
        DATA_START = args.data_start
    if args.data_limit:
        DATA_LIMIT = args.data_limit

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()

    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Starting test on device" + str(model.device))
    # dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    glove_model300 = api.load('glove-wiki-gigaword-300')

    # evaluate(args, model, tokenizer, test_data, prefix="test")
    test_data = read_json_data(args.input_file)
    evaluate(args, model, glove_model300, tokenizer, test_data, prefix="train_simt2")
    ##for lading directly from csv
    # dataset = load_dataset('csv', data_files="../../data_prep/daily_dialogue_act/data/test_da_key.csv", split='train[:20%]')
    ##for loading from csv into pandas and aggregating first 
    # test_df = pd.read_csv(test_file)
    # test_df = aggregate_rows(test_df)
    # dataset = Dataset.from_pandas(test_df)


    exit(0)




if __name__ == '__main__':
    try:
        set_start_method('spawn')
        # torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
    except RuntimeError:
        pass
    run()
