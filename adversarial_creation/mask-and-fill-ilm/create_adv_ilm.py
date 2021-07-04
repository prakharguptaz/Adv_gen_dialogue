from enum import Enum
from collections import defaultdict
import multiprocessing
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, CONFIG_NAME, WEIGHTS_NAME
from collections import Counter
import os
import random
import importlib
import pickle
import sys
import re
import torch
import ilm.constants
import ilm.mask
import ilm.mask.util
import ilm.tokenize_util 
from ilm.datasets import Dataset, get_dataset
import ilm.mask
from ilm.mask.util import mask_cls_str_to_type
from ilm.mask.util import apply_masked_spans
from ilm.mask.util import masked_spans_bounds_valid, masked_spans_overlap
from create_ilm_examples import randomly_mask_dataset, randomly_mask_document
import ilm.tokenize_util
from ilm.infer import infill_with_ilm, infill_naive_with_ilm, infill_function_withexp
from torch.multiprocessing import Pool, Process, set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
##https://srijithr.gitlab.io/post/pytorchdist/
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from transformers import GPT2LMHeadModel
import json
import re
import pprint
import logging
import requests
import math
import re
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
#https://www.scribendi.ai/comparing-bert-and-gpt-2-as-language-models-to-score-the-grammatical-correctness-of-a-sentence/
#https://github.com/simonepri/lm-scorer
#https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
#https://srijithr.gitlab.io/post/pytorchdist/

#to be set with args
DATA_LIMIT=None
DATA_START=None

MODEL_DIR = 'train'
tokenizer = ilm.tokenize_util.Tokenizer.GPT2
mask_type = mask_cls_str_to_type('ilm.mask.hierarchical.MaskHierarchical')
masker = mask_type(0.25, mask_paragraph_p=0.0, mask_document_p=0.0)


tokenizer = ilm.tokenize_util.Tokenizer.GPT2
with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
    additional_ids_to_tokens = pickle.load(f)
additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
print(ilm.tokenize_util.encode('no , i dont<|infill_word|><|infill_word|> .<|infill_sentence|>', tokenizer))
try:
    lenvocab = ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
except ValueError:
    print('Already updated')
print(ilm.tokenize_util.encode('no , i dont<|infill_word|><|infill_word|> .<|infill_sentence|>', tokenizer))
print(additional_tokens_to_ids)

n_gpu = torch.cuda.device_count()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
# scorer_device = "cuda" if torch.cuda.is_available() else "cpu"# + str(n_gpu-1) if torch.cuda.is_available() else "cpu"
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
scorer_device = "cuda:" + str(n_gpu-1) if torch.cuda.is_available() else "cpu"
batch_size_scorer = 2
scorer = LMScorer.from_pretrained("gpt2", device=scorer_device, batch_size=batch_size_scorer)

error_to_count_total = Counter()
num_examples_per_document=4
max_num_retries=5

parallel = False

# if parallel:
#   scorer.share_memory()

stop_words = ['another', 'once', 'thereupon', 'whom', 'regarding', 'first', 'anyhow', 'whence', 'else', 'might', 'themselves', '’s', 'of', 'side', "'m", 'top', 'ours', 'will', 'whole', 'sixty', 'if', 'but', 'serious', 'cannot', 'became', 'about', 'do', 'take', 'an', 'being', 'throughout', 'after', 're', 'were', 'must', 'thence', 'whether', 'whereafter', 'hundred', 'again', 'still', 'further', 'above', 'third', 'them', 'any', 'why', "'ll", 'wherein', 'towards', 'every', 'five', 'most', 'into', 'meanwhile', 'may', 'onto', 'neither', 'namely', 'fifteen', 'i', 'below', 'they', 'without', 'him', 'never', 'give', 'forty', 'own', 'thus', 'whereby', 'yourself', 'itself', 'somewhere', 'via', 'full', 'next', 'been', 'always', 'put', 'whereupon', 'because', 'so', 'under', 'during', 'than', 'several', 'upon', 'very', '’d', 'something', "n't", 'ten', '‘m', 'though', 'anything', 'fifty', 'all', 'seemed', 'well', 'twenty', 'more', 'amongst', 'wherever', 'name', 'am', 'therein', 'much', 'among', 'less', 'when', 'except', 'hereafter', 'has', 'along', 'seems', 'now', 'up', 'sometimes', 'alone', 'ca', 'everything', 'enough', 'himself', 'everyone', '‘ve', 'quite', '‘re', 'elsewhere', 'whoever', 'it', 'back', 'me', 'otherwise', 'perhaps', 'latter', 'on', 'already', 'across', 'whither', 'what', 'within', '‘d', 'n’t', "'ve", 'that', 'nevertheless', 'someone', 'nowhere', 'empty', 'out', 'some', 'really', 'off', 'each', 'mostly', 'hence', 'yet', 'are', 'using', 'nothing', 'yourselves', 'no', "'re", 'besides', 'over', '‘ll', 'sometime', 'becomes', 'before', 'anywhere', 'by', 'seem', 'for', 'us', 'where', 'many', 'these', 'he', 'toward', 'her', 'should', 'doing', 'ever', 'nor', 'three', 'between', 'can', 'same', 'whereas', 'until', 'either', 'their', 'due', '’ll', 'beside', 'few', 'the', 'was', 'which', 'its', 'just', 'our', 'your', 'say', 'noone', 'front', 'against', 'down', 'such', 'anyway', 'also', 'everywhere', 'two', 'together', 'and', 'others', 'bottom', 'eight', 'we', 'my', "'d", 'whatever', 'six', 'indeed', 'did', 'other', 'becoming', 'afterwards', 'from', 'thereafter', 'too', 'you', 'behind', 'mine', 'a', 'thereby', 'not', 'to', 'nobody', 'be', 'done', 'then', 'at', 'even', '’re', 'here', 'various', 'make', 'twelve', 'how', 'as', 'since', 'there', 'call', 'somehow', 'she', 'in', 'anyone', 'almost', 'moreover', 'beyond', 'herself', 'yours', 'hereby', 'both', 'nine', 'latterly', 'herein', 'have', 'eleven', 'while', 'his', 'please', 'n‘t', 'move', 'those', 'get', 'could', 'beforehand', 'this', 'is', 'per', 'although', 'hers', 'made', '’m', 'often', 'ourselves', 'therefore', 'whose', 'keep', 'only', 'none', 'seeming', 'one', '’ve', 'hereupon', 'whenever', 'unless', '‘s', 'does', 'had', 'would', 'however', 'formerly', 'see', 'used', 'show', 'around', 'part', "'s", 'become', 'least', 'thru', 'last', 'who', 'rather', 'myself', 'through', 'former', 'four', 'or', 'with', 'go']
stop_words_small = ["I", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

context_test_string = """<|context|> <|speaker1|> while the curtains are being made , i can start having people look at the kitchen . i can't stand that old kitchen . i won't\
be able to cook there . i don't want to use that electric stove . <|speaker2|> we need to find an interior decorating company to redecorate the kitchen . i believe in portland there are shops that specialize in kitchen renovation . i will look in the yellow pages .\
i'd like a kitchen mostly in ivory and light green . <|speaker1|> i agree . the colors must be soft and pleasant . you should feel comfortable when you cook our dinners . <speaker2>\
me ? cook our dinners ? hah ! you will be cooking , dear . you will cook . <|end of context|> <|response|> <|speaker2|> no , i don't _ _ . _""".strip()


import subprocess as sp

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

get_gpu_memory()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()



p = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+))')
def cap(match):
    return(match.group().capitalize())

def read_json_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
            
    return data

def get_context_string(doc, history_len = 4):
    context_string = '<|context|>'
    sents = doc['context'][-history_len:]
    for i,sent in enumerate(sents):
        context_string += ' <|speaker' + str((i-(len(sents)+1)%2)%2 +1) + '|> ' + sent
    context_string += ' <|endofcontext|> <|response|> <|speaker2|> '
#     doc = doc['response'][:]
    
    return context_string



def test_infill(model, tokenizer, additional_tokens_to_ids):
  context_ids = ilm.tokenize_util.encode(context_test_string, tokenizer)
  # Replace blanks with appropriate tokens from left to right
  _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]
  context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
  context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
  context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
  print(ilm.tokenize_util.decode(context_ids, tokenizer))
  # print(context_ids)
  print(ilm.tokenize_util.encode('no , i dont<|infill_word|><|infill_word|> .<|infill_sentence|>', tokenizer))
  test_s = (ilm.tokenize_util.decode(context_ids, tokenizer))
  # print(test_s)
  etest_s =ilm.tokenize_util.encode(test_s, tokenizer)
  # print(etest_s)
  test_ss = (ilm.tokenize_util.decode(etest_s, tokenizer))
  # print(context_ids)
  # print(etest_s)
  # print(test_s)
  print(test_ss)
  print('begin\n')
  generated = infill_with_ilm(
      model,
      additional_tokens_to_ids,
      context_ids,
      num_infills=2)
  for g in generated:
      print('-' * 80)
      print(ilm.tokenize_util.decode(g, tokenizer))

num_retries_total = 0
def get_masked_spans(docs):
  result = []
  docs_masked=[]
#     for doc in tqdm(docs):
  for doc in (docs):
    # print(doc)
    doc_masks, error_to_count = randomly_mask_document(
        doc,
        masker,
        num_examples_per_document,
        max_num_retries=max_num_retries,
          min_masked_spans=None,
          max_masked_spans=None,
          random_sample_down_to_max=True,
          ensure_valid_bounds_in_spans=True,
          ensure_nonoverlapping_spans=True,
        )
    # print(doc_masks, error_to_count)
    docs_masked.append((doc, doc_masks))
    for k, v in error_to_count.items():
        error_to_count_total[k] += v
#     print(len(docs_masked[0]))
  i = 0
  for doc, examples in docs_masked:
#         if len(examples) == 0:
#             continue
      for masked_spans in examples:
          mask_span_type_to_str = {t:'<|{}|>'.format(str(t)) for t, _, _ in masked_spans}
          context, answers = apply_masked_spans(
              doc,
              masked_spans,
              mask_span_type_to_str)
          if context in ['<|MaskHierarchicalType.DOCUMENT|>', '<|MaskHierarchicalType.PARAGRAPH|>', '<|MaskHierarchicalType.SENTENCE|>']:
            continue
          result.append([doc,context, answers])

  return result



def get_infill_substitutes(sentence):
#     '<|infill_document|>': 50259, '<|infill_paragraph|>': 50260, '<|infill_sentence|>': 50261, '<|infill_ngram|>': 50262, '<|infill_word|>
    sentence = sentence.replace('<|MaskHierarchicalType.PARAGRAPH|>', '<|infill_paragraph|>')
    sentence = sentence.replace('<|MaskHierarchicalType.DOCUMENT|>', '<|infill_document|>')
    sentence = sentence.replace('<|MaskHierarchicalType.SENTENCE|>', '<|infill_sentence|>')
    sentence = sentence.replace('<|MaskHierarchicalType.NGRAM|>', '<|infill_ngram|>')
    sentence = sentence.replace('<|MaskHierarchicalType.WORD|>', '<|infill_word|>')

    return sentence

_blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]

def get_context_ids(data_input):
    
    types_of_blanks = ['<|infill_document|>', '<|infill_paragraph|>', '<|infill_sentence|>', '<|infill_ngram|>', '<|infill_word|>']
    
    list_of_blanks = []
    s = data_input
    while True:
        m = re.search('\<\|infill_(.+?)\|\>',s)
        if m:
            found = m.group(1)
            list_of_blanks.append('<|infill_'+found+'|>')
            s = s.replace('<|infill_'+found+'|>', ' _',1)
        else:
            break
    data_input = s
    
    context_ids = ilm.tokenize_util.encode(data_input, tokenizer)
    for type_blank in list_of_blanks:
        context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids[type_blank]

    return context_ids

def test_pure_infill(infill_sent):
    infills = re.findall('\<\|infill_(.+?)\|\>',infill_sent)
    words = infill_sent.split()
#     print(len(infills), len(words))
    if len(words)<=len(infills):
        return True
    
    if len(infills)<=1:
        if len(infills)==1 and infills[0] in ['<|infill_ngram|>', '<|infill_word|>']:
            return True
        if len(infills)==0: return True
    
    
    return False

def get_avoidtokens(tokens_eachblank, tokenizer):
    tokens_eachblank = [' '.join([y.replace('?', '') for y in x.split() if len(y)>2]) for x in tokens_eachblank]
    all_avoids = []
    for x in tokens_eachblank:
        xtokens_list = []
        for y in x.split():
            y = y.replace('?', '').replace('.', '').replace('!', '').replace(',', '')
            if len(y)<3: continue
            try:
                tt = ilm.tokenize_util.tokens_to_ids([y])
                xtokens_list+=(tt)
            except: pass
            try:
                st = ilm.tokenize_util.tokens_to_ids([' '+y])
                xtokens_list+=(st)
            except:pass
        all_avoids.append(xtokens_list)
                
    avoid_tokens = [ilm.tokenize_util.tokenize(x) for x in tokens_eachblank]
    avoid_ids = [ilm.tokenize_util.tokens_to_ids(x) for x in avoid_tokens]
    avoid_ids = [list(set(a).union(set(b))) for a,b in zip(avoid_ids, all_avoids)]
    # tokens_eachblank = [' '.join([y.replace('?', '') for y in x.split() if len(y)>2]) for x in tokens_eachblank]
    return avoid_tokens, avoid_ids

def get_infilled_responses(model, tokenizer, additional_tokens_to_ids, context, sentence_infill_pairs, num_infills=1, verbose=False):
    list_generated_responses = []
    for o in sentence_infill_pairs:
        infill_sent = get_infill_substitutes(o[1])
        is_high_infill = test_pure_infill(infill_sent)
        if is_high_infill:
            continue
        if verbose: print('infill_sent:', infill_sent)
        if type(context)==list: 
          context_to_use = random.choice(context)
        else:
          context_to_use = context
        data_input = context_to_use + ' ' + infill_sent
        context_ids = get_context_ids(data_input)

        generated = infill_with_ilm(
            model,
            additional_tokens_to_ids,
            context_ids,
            num_infills=num_infills)
        for g in generated:
            generated_sent = ilm.tokenize_util.decode(g, tokenizer)
            generated_response = generated_sent.split('<|response|> <|speaker2|> ')[-1]
#             print('-' * 80)
#             print(generated_sent)
            if verbose: print('GENERATED:', generated_response)
            list_generated_responses.append(generated_response)
        if verbose: print()
#     random.shuffle(list_generated_responses)
    
    return list_generated_responses


def get_infilled_responses_avoidtokens(model, tokenizer, additional_tokens_to_ids, context, sentence_infill_pairs, num_infills=1, verbose=False):
    '''Version which avoid original words which were blanked while generating infilled responses. It is a bit slow. get_infilled_responsesv1 works faster'''
    
    list_generated_responses = []
    for o in sentence_infill_pairs:
        infill_sent = get_infill_substitutes(o[1])
        is_high_infill = test_pure_infill(infill_sent)
        if is_high_infill:
            continue
        if verbose: print(infill_sent)
        if type(context)==list: 
          context_to_use = random.choice(context)
        else:
          context_to_use = context
        data_input = context_to_use + ' ' + infill_sent
        # print(o)
        # import pdb;pdb.set_trace()
        context_ids = get_context_ids(data_input)
        tokens_eachblank = [x[1] for x in o[2]]
        avoid_tokens, avoid_ids = get_avoidtokens(tokens_eachblank, tokenizer)
        avoid_ids = [[]]
        # print(avoid_tokens, avoid_ids)
        generated = infill_function_withexp(
        # generated = infill_with_ilm(
            model,
            additional_tokens_to_ids,
            context_ids,
            avoid_ids,
            num_infills=num_infills)
        for g in generated:
            # print(g)
            generated_sent = ilm.tokenize_util.decode(g, tokenizer)
            generated_response = generated_sent.split('<|response|> <|speaker2|> ')[-1]
#             print('-' * 80)
#             print(generated_sent)
            if verbose: print('GENERATED RESPONSE: ', generated_response)
            list_generated_responses.append(generated_response)
        if verbose: print()
#     random.shuffle(list_generated_responses)
    
    return list_generated_responses




def get_infilled_responses_naive(model, tokenizer, additional_tokens_to_ids, context, sentence_infill_pairs, num_infills=2):
    list_generated_responses = []
    for o in sentence_infill_pairs:
        infill_sent = get_infill_substitutes(o[1])
#         print(infill_sent)
        is_high_infill = test_pure_infill(infill_sent)
        if is_high_infill:
            continue
#         print(o[1])
#         print(infill_sent)
        data_input = context + ' ' + infill_sent
        context_ids = get_context_ids(data_input)
#         print(data_input)
        
        generated = infill_naive_with_ilm(
            model,
            additional_tokens_to_ids,
            context_ids,
            num_infills=num_infills)
        for g in generated:
            generated_sent = ilm.tokenize_util.decode(g, tokenizer)
            generated_response = generated_sent.split('<|startofinfill|>')[-1]
#             print('-' * 80)
#             print(generated_sent)
#             print(generated_response)
            if generated_response=='': continue
            list_generated_responses.append(generated_response)
#         print()
#     random.shuffle(list_generated_responses)
    
    return list_generated_responses


def test_random_infill(model, tokenizer, additional_tokens_to_ids):
  docs_in = [' we need to find an interior decorating company to redecorate the kitchen',' This a trial response.', 'Now we should fill in the response well.','i believe in portland there are shops that specialize in kitchen renovation .']
  outs = get_masked_spans(docs_in)
  print('Test mask spans')
  print(outs)

  print('Test infilled responses')
  infilled_responses = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, context_test_string[:-21], outs, verbose=True)
  print(infilled_responses)


def test_random_infill_parallel(model, tokenizer, additional_tokens_to_ids):
  docs_in = [' we need to find an interior decorating company to redecorate the kitchen']#,' This a trial response.', 'Now we should fill in the repsonse well.',' . i believe in portland there are shops that specialize in kitchen renovation .']
  outs = get_masked_spans(docs_in)
  print('Test mask spans')
  print(outs)


  to_process = []
  # for data_point in tqdm(data_points, desc="Evaluating"):
  # data_points = data_points[:100]
  for i, data_point in enumerate(outs):
      # history, utterance, frames, tokenizer = data_point
      # print([outs[i]])
      to_process.append([i, model, tokenizer, additional_tokens_to_ids, context_test_string[:-21], [outs[i]]])

  pool = Pool(2)
  result_objects = [pool.apply_async(map_to_sample, args=(a)) for a in to_process]
  results = [r.get() for r in result_objects]
  # result_objects = [(map_to_sample(*a)) for a in to_process]
  # results = [r for r in result_objects]
  results.sort(key=lambda x: x[0])

  print('Test infilled responses')
  for i, (result, data_point) in enumerate(zip(results, outs)):
      ind, text = result
      # print(ind, data_point)
      print(result)

  # infilled_responses = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, context_test_string[:-21], outs)
  # print(infilled_responses)


def map_to_sample(i, model, tokenizer, additional_tokens_to_ids, context_test_string, outs):
    text = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, context_test_string, outs)
    if i%100==0:
        print(i)

    return i, text


def get_tokenizer(MODEL_DIR):
  tokenizer = ilm.tokenize_util.Tokenizer.GPT2
  with open(os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl'), 'rb') as f:
      additional_ids_to_tokens = pickle.load(f)
  additional_tokens_to_ids = {v:k for k, v in additional_ids_to_tokens.items()}
  print(ilm.tokenize_util.encode('no , i dont<|infill_word|><|infill_word|> .<|infill_sentence|>', tokenizer))
  try:
      lenvocab = ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
  except ValueError:
      print('Already updated')
  print(ilm.tokenize_util.encode('no , i dont<|infill_word|><|infill_word|> .<|infill_sentence|>', tokenizer))
  print(additional_tokens_to_ids)

  return tokenizer, additional_tokens_to_ids




def check_ispoor_text(sentence):
  sentence_words = sentence.split()
  
  for i, word in enumerate(sentence_words):
      if i>1 and word==sentence_words[i-1]:
          return True
      
  return False

def get_lm_ppl(sentences, scorer):
    sentences = [p.sub(cap, r) for r in sentences]
    scores = scorer.sentence_score(sentences, log=True)
    scores_charlength = [s / (len(r)+2) for s,r in zip(scores, sentences)]
    scores_length = [s / (len(r.split())+2) for s,r in zip(scores, sentences)]

    return list(map(lambda x, y, z:(x,y,z), sentences, scores_charlength, scores_length)) 


def filter_clean_adv_set(scorer, adv_gen_neg_responses_t1, positive_responses, desired_number_responses = 5):
    
    #cleanup responses
    adv_gen_neg_responses = []
    for i, response in enumerate(adv_gen_neg_responses_t1):
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
#         clean_response = re.sub(r'($)?\s+(\d)(.)?\s+(\d)', r'\1\2', clean_response)
        adv_gen_neg_responses.append(clean_response)
    
    adv_gen_neg_responses_t1 = adv_gen_neg_responses
    
    positive_responses_wordset_list = []
    for pr in positive_responses:
        positive_responses_wordset_list.append(set(pr.split()))
    
    #get lm scores 
    lm_scores = get_lm_ppl(adv_gen_neg_responses_t1, scorer)
    lm_scores, adv_gen_neg_responses_t1 = zip(*sorted(zip(lm_scores, adv_gen_neg_responses_t1),key=lambda x:x[0][2]+x[0][1],reverse=True))
#     print(lm_scores)
    good_responses = []
    responses_words_list = []
    threshold = 0.1
    sent_length_threshold = 6
    tries = 0
    while True:
        tries+=1
#         print('looped')
        for i, response in enumerate(adv_gen_neg_responses_t1):
            words_set = set(response.split())
            max_sim_score = -0.1
            #if sentence too small, decrease its chances
            if len(words_set)<=sent_length_threshold:
                max_sim_score = 0.89
                
            for old_response_words_set in responses_words_list:
                intersection_words = old_response_words_set.intersection(words_set)
                score = len(intersection_words)/(2*len(words_set)+1)
                max_sim_score = max(score, max_sim_score)
                # new_words = words_set - old_response_words_set - set(stop_words_small)
                # if len(new_words) <= 1:
                #     max_sim_score =0.89

            #should not have too much overlap with positive responses
            for pos_response_words_set in positive_responses_wordset_list:
                # interesection_words = pos_response_words_set.intersection(words_set)
                # if len(interesection_words) >= len(pos_response_words_set)-1:
                #     max_sim_score =0.89
                new_words = words_set - pos_response_words_set - set(stop_words_small)
                if len(new_words) <= 2:
                    max_sim_score =0.49
#                 score = len(interesection_words)/len(words_set) 
#                 if score >=0.8:
#                 print(pos_response_words_set, words_set, score)

            if response in positive_responses or response in good_responses:
                max_sim_score =0.90
            
            ##remove poor text, todo: add language model score
            #if check_ispoor_text(response):
            lm_sent = lm_scores[i][1]
            lm_sent_wordlevel = lm_scores[i][2]
#             print(lm_scores[i])
            if lm_sent<-1.4 or lm_sent_wordlevel<-4.9:
                max_sim_score = 0.89
                     
                
            if max_sim_score==-0.1 or max_sim_score<threshold:
#                 print('--addingresponse ', response, max_sim_score)
                good_responses.append(response)
                responses_words_list.append(words_set)
                
        threshold+=0.1
        if sent_length_threshold>2:
            sent_length_threshold-=1
        
        if len(good_responses)>=desired_number_responses:
            return good_responses[:desired_number_responses]
        
        if tries>4 or threshold>0.99:
            print('choosing randomly, got ', len(good_responses))
#             print(good_responses,list(adv_gen_neg_responses_t1))
#             return random.sample(good_responses+list(adv_gen_neg_responses_t1), desired_number_responses)
            return good_responses
        
#         print(good_responses)
    good_responses = good_responses + adv_gen_neg_responses_t1
    return good_responses[:desired_number_responses]


def get_maskable_candidates(context, positive_responses):
    candidates = []
    reserve = []
    for pr in positive_responses:
        prtokens = pr.lower().split()
        prtokens = [t for t in prtokens if t.replace('.', '') not in stop_words]
#         print(prtokens)
        if len(prtokens)<2:
            reserve.append(pr)
        else:
            candidates.append(pr)
    
    for pr in context:
        prtokens = pr.lower().split()
        prtokens = [t for t in prtokens if t not in stop_words]
        if len(prtokens)<2:
            reserve.append(pr)
        else:
            candidates.append(pr)
    
#     print(candidates, reserve)
    return candidates + reserve


def get_maskable_candidates_contextfirst(context, positive_responses):
    candidates = []
    reserve = []

    for pr in context:
        prtokens = pr.lower().split()
        prtokens = [t for t in prtokens if t not in stop_words]
        if len(prtokens) < 2:
            reserve.append(pr)
        else:
            candidates.append(pr)

    for pr in positive_responses:
        prtokens = pr.lower().split()
        prtokens = [t for t in prtokens if t.replace('.', '') not in stop_words]
        #         print(prtokens)
        if len(prtokens) < 2:
            reserve.append(pr)
        else:
            candidates.append(pr)

    #     print(candidates, reserve)
    return candidates + reserve


def add_modgt_responses_v1(model, additional_tokens_to_ids, test_data_point, train_data, verbose=False, num_negative_candidates=10):
    positive_responses = test_data_point['positive_responses']
    context = test_data_point['context']
    context_string = test_data_point['context_string']
    # print(positive_responses)
    masked_spans = []
    # candidates = get_maskable_candidates(context, positive_responses)
    candidates = get_maskable_candidates_contextfirst(context, positive_responses)

    candidates = candidates[:num_negative_candidates]
    for pr in candidates:
        masked_spans += get_masked_spans([pr])
    # print(masked_spans)
    random_train = random.choice(train_data)
    random_context = random_train['context']
    random_context_string = random_train['context_string']
    if verbose: print('random_context_string ',random_context_string)
#     print('random_context_string ',random_context_string)
    adv_gen_neg_responses_t1 = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, random_context_string, masked_spans, num_infills=3, verbose=verbose)
#     adv_gen_neg_responses_t1 = get_infilled_responses_naive(model, tokenizer, additional_tokens_to_ids, random_context_string, masked_spans, num_infills=2)
    if verbose: print(adv_gen_neg_responses_t1)

    adv_gen_neg_responses_t1 = filter_clean_adv_set(scorer, adv_gen_neg_responses_t1, positive_responses, desired_number_responses=num_negative_candidates)
    test_data_point['adv_gen_neg_responses_t1'] = adv_gen_neg_responses_t1 

    return adv_gen_neg_responses_t1


def add_modgt_responses(model, additional_tokens_to_ids, test_data_point, train_data, verbose=False, desired_number_responses=20):
    positive_responses = test_data_point['positive_responses']
    context = test_data_point['context']
    context_string = test_data_point['context_string']
    #print(positive_responses)
    masked_spans = []
    context_c = [x for i,x in enumerate(context) if (len(context)-i+1)%2==0]
    candidates = get_maskable_candidates(context_c, positive_responses)
    
    retrieved_responses_sampled = test_data_point['bm25_sampled']
    retrieved_responses = []
    for rs in retrieved_responses_sampled:
        if len(rs.split())>7 and rs not in retrieved_responses and rs not in positive_responses:
            retrieved_responses.append(rs)
    candidates_retrieved_responses = get_maskable_candidates([], retrieved_responses)        
    candidates+=candidates_retrieved_responses

    candidates = candidates[:desired_number_responses]
    for pr in candidates:
        masked_spans += get_masked_spans([pr])
    random.shuffle(masked_spans)
    masked_spans = masked_spans[:desired_number_responses+12]

    random_context_strings = []
    while len(random_context_strings)<desired_number_responses:
      random_train = random.choice(train_data)
      random_context = random_train['context']
      random_context_string = random_train['context_string']
      random_context_strings.append(random_context_string)
      if verbose: print('random_context_string ',random_context_string)
    # adv_gen_neg_responses_t1 = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, random_context_string, masked_spans, num_infills=5, verbose=verbose)
    adv_gen_neg_responses_t1 = get_infilled_responses(model, tokenizer, additional_tokens_to_ids, random_context_strings, masked_spans, num_infills=5, verbose=verbose)

#     print(positive_responses)
    if verbose: print(adv_gen_neg_responses_t1)
#     print('now filter')
    adv_gen_neg_responses_t1 = filter_clean_adv_set(scorer, adv_gen_neg_responses_t1, context+positive_responses,desired_number_responses=desired_number_responses)
    adv_gen_neg_responses_t1+=test_data_point['random_negative_responses']
    while len(adv_gen_neg_responses_t1)<desired_number_responses:
      random_train_dp = random.choice(train_data)
      adv_gen_neg_responses_t1+=random_train_dp['random_negative_responses']
    adv_gen_neg_responses_t1 = adv_gen_neg_responses_t1[:desired_number_responses]
    test_data_point['adv_gen_neg_responses_t1'] = adv_gen_neg_responses_t1 

    return adv_gen_neg_responses_t1
    

def get_data(data_path):
  #original
  test_data = read_json_data(data_path + 'sample_file.json')
  train_data = read_json_data(data_path + 'sample_file.json')
  dev_data = read_json_data(data_path + 'sample_file.json')
  #with retrieved
  # test_data = read_json_data(data_path + 'dailydialog_orig/anserini_valid/test_dd_ns20.json')
  # train_data = read_json_data(data_path + 'dailydialog_orig/anserini_valid/train_dd_nsall20.json')
  # dev_data = read_json_data(data_path + 'dailydialog_orig/anserini_valid/dev_dd_ns20.json')

  for data in dev_data:
      data['context_string'] = get_context_string(data)
  for data in test_data:
      data['context_string'] = get_context_string(data)
  for data in train_data:
      data['context_string'] = get_context_string(data)

  return train_data, test_data, dev_data 


def save_data_to_disk(data, dest_file):
  output_file = open(dest_file, 'w', encoding='utf-8')
  for i,dic in enumerate(data):
      # if DATA_START and i<DATA_START: continue
      # if DATA_LIMIT and i==DATA_LIMIT:break
      json.dump(dic, output_file) 
      output_file.write("\n")


def add_adv(model, scorer, tokenizer, additional_tokens_to_ids, data, args, data_type):
  dest_file = args.dest_path + data_type + '_' + args.experiment_name + '.json' 
  print(dest_file)
  data_to_process = []
  for i,data_point in tqdm(enumerate(data)):
      if DATA_START and i<DATA_START: continue
      # if i%100==0:
      #     print(i)
      if DATA_LIMIT and i==DATA_LIMIT:break
      # print(i)
  #     positive_responses = test_data['positive_responses']
      add_modgt_responses(model, additional_tokens_to_ids, data_point, data)
      data_to_process.append(data_point)

  #     pprint.pprint(data_point, indent=1)

  save_data_to_disk(data_to_process, dest_file)

def add_adv_parallel(model, additional_tokens_to_ids, data, args, data_type):
  dest_file = args.dest_path + data_type + '_' + args.experiment_name + '.json' 
  print(dest_file)

  to_process = []
  # for data_point in tqdm(data_points, desc="Evaluating"):
  # data_points = data_points[:100]
  data_to_process = []
  for i, data_point in enumerate(data):
    if DATA_START and i<DATA_START: continue
    if DATA_LIMIT and i==DATA_LIMIT:break
    data_to_process.append(data_point)
      # history, utterance, frames, tokenizer = data_point
      # print([outs[i]])
    to_process.append([i, model, additional_tokens_to_ids, data_point, data])

  print(len(data_to_process), ' : data points to process')
  pool = Pool(4)
  result_objects = [pool.apply_async(map_to_gen, args=(a)) for a in to_process]
  # result_objects = [(map_to_sample(*a)) for a in to_process]
  # results = [r.get() for r in result_objects]
  results = []
  for job in tqdm(result_objects):
        results.append(job.get())
  # print(results)
  results.sort(key=lambda x: x[0])

  print('infilled responses completed')
  for i, (result, data_point) in enumerate(zip(results, data_to_process)):
      ind, text = result
      data_point['adv_gen_neg_responses_t1'] = text
      # print(ind, data_point)
      # print(result)


  print(len(data_to_process), ' : data points to save')

  save_data_to_disk(data_to_process, dest_file)
  
    

def map_to_gen(i, model, additional_tokens_to_ids, data_point, data):
    text = add_modgt_responses(model, additional_tokens_to_ids, data_point, data)
    if i%100==0:
        print(i)
    # print(i)

    return i, text


def get_model_tokenizer(args):
  # MASK_CLS = args.mask_cls
  # MASK_PROB = args.mask_prob
  # mask_type = mask_cls_str_to_type(MASK_CLS)
  # global masker
  # masker = mask_type(MASK_PROB)

  MODEL_DIR = args.model_dir
  model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
  model.eval()
  _ = model.to(device)


  if parallel:
    model.share_memory()

  tokenizer, additional_tokens_to_ids = get_tokenizer(MODEL_DIR)

  return model, scorer, tokenizer, additional_tokens_to_ids


def create(args):
  # MASK_CLS = args.mask_cls
  # MASK_PROB = args.mask_prob
  # mask_type = mask_cls_str_to_type(MASK_CLS)
  # global masker
  # masker = mask_type(MASK_PROB)

  MODEL_DIR = args.model_dir
  model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
  model.eval()

  # if torch.cuda.device_count() > 1:
  #     print("We have available ", torch.cuda.device_count(), "GPUs!")
  #     model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
  
  # tokenizer, additional_tokens_to_ids = get_tokenizer(MODEL_DIR)


  if not parallel:
    _ = model.to(device)
    train_data, test_data, dev_data = get_data(args.data_path)
    add_adv(model, scorer, tokenizer, additional_tokens_to_ids, test_data, args, 'test')
    # add_adv(model, scorer, tokenizer, additional_tokens_to_ids, dev_data, args, 'dev')
    # add_adv(model, scorer, tokenizer, additional_tokens_to_ids, train_data, args, 'train')
  else:
    
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)
    # print('local rank', args.local_rank, device, 'n_gpu ', n_gpu)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    _ = model.to(device)
    print('Putting on device', device)
    model.share_memory()
    train_data, test_data, dev_data = get_data(args.data_path)
    # add_adv_parallel(model, additional_tokens_to_ids, test_data, args, 'test')
    # print('test done')
    # add_adv_parallel(model, additional_tokens_to_ids, dev_data, args, 'dev')
    # print('dev done')
    add_adv_parallel(model, additional_tokens_to_ids, train_data, args, 'train')

  exit(0)

def create_dist(rank, args, world_size, use_cuda):
  print(f"Running basic DDP example on rank {rank}.")
  setup(rank, world_size)
  device = torch.device("cuda" if use_cuda else "cpu")

  MODEL_DIR = args.model_dir
  model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
  model.eval()

  model  = model.to(rank)
  model = DDP(model, device_ids=[rank])

  train_data, test_data, dev_data = get_data(args.data_path)
  add_adv(model, scorer, tokenizer, additional_tokens_to_ids, test_data, args, 'test')
  # add_adv(model, scorer, tokenizer, additional_tokens_to_ids, dev_data, args, 'dev')
  # add_adv(model, scorer, tokenizer, additional_tokens_to_ids, train_data, args, 'train')

  cleanup()





def main():

  from argparse import ArgumentParser

  parser = ArgumentParser()

  parser.add_argument('--experiment_name', type=str)
  # parser.add_argument('train_dir', type=str)
  # parser.add_argument('examples_dir', type=str)
  parser.add_argument('--data_path', type=str, default='../../dataset/')
  parser.add_argument('--dest_path', type=str, default='../../dataset/')
  parser.add_argument('--model_dir', type=str, default='train')
  parser.add_argument('--data_limit', type=int)
  parser.add_argument('--data_start', type=int)
  parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
  parser.add_argument('--gpus', type=int, default=1, metavar='N',
                        help='Number of GPUs')

  mask_args = parser.add_argument_group('Mask')
  mask_args.add_argument('--mask_cls', type=str)
  mask_args.add_argument('--mask_prob', type=float, default=0.25)

  # tokenizer_args = parser.add_argument_group('Tokenizer')
  # tokenizer_args.add_argument('--tokenizer_name', type=str, choices=[t.name.lower() for t in ilm.tokenize_util.Tokenizer])
  # tokenizer_args.add_argument('--tokenizer_custom_vocab_fp', type=str)

  # data_args = parser.add_argument_group('Data')
  # data_args.add_argument('--data_no_cache', action='store_false', dest='data_cache')
  # data_args.add_argument('--data_loader_num_workers', type=int)


  parser.set_defaults(
      seed=None,
      wandb=False,
      wandb_project_name='ilm',
      mask_cls='ilm.mask.hierarchical.MaskHierarchical',
      # mask_cls=0.25,
      tokenizer_name='gpt2',
      tokenizer_custom_vocab_fp=None,
      task='ilm',
      data_cache=True,
      data_loader_num_workers=4,
      model_name='gpt2')
  
  args = parser.parse_args()


  global DATA_LIMIT
  global DATA_START
  if args.data_start:
    DATA_START = args.data_start
  if args.data_limit:
    DATA_LIMIT = args.data_limit

  if args.seed is None:
    args.seed = random.randint(0, 1e6)
  print('Random seed {}'.format(args.seed))

  use_cuda = torch.cuda.is_available()
  world_size = args.gpus
  if torch.cuda.device_count() > 1:
    print("We have available ", torch.cuda.device_count(), "GPUs! but using ",world_size," GPUs")
  #########################################################
  # mp.spawn(create_dist, args=(args, world_size, use_cuda), nprocs=world_size, join=True)  
  model, scorer, tokenizer, additional_tokens_to_ids = get_model_tokenizer(args)
  model = GPT2LMHeadModel.from_pretrained(args.model_dir)
  model.eval()
  model = model.to(device)

  test_random_infill(model, tokenizer, additional_tokens_to_ids)
  # test_random_infill_parallel(model, tokenizer, additional_tokens_to_ids)

  create(args)


if __name__ == '__main__':
  try:
      set_start_method('spawn')
  except RuntimeError:
      pass
  main()

#python dd_train_ilm.py experiment_dd train data/char_masks/dailydialog/ --seed 0 --train_examples_tag train --eval_examples_tag valid --eval_max_num_examples 512 --mask_cls ilm.mask.hierarchical_dailydialog.MaskHierarchical
#python create_adv_ilm_cf6.py --experiment_name trr --data_start 0 --data_limit 10

#Note set paralllel = True if multigpu is needed through threading
#Note if there is a need to use techniques to avoid infilling originally blanked words, use the get_infilled_responses_avoidtokens function but it is slow
