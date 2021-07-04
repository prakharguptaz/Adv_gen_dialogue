import copy

import torch
import torch.nn.functional as F


def sample_from_logits(
    logits,
    temp=1.,
    topk=None,
    nucleus=1.):
  if temp == 0:
    return torch.argmax(logits, dim=-1).unsqueeze(-1)
  elif temp != 1:
    logits /= temp
  
  probs = F.softmax(logits, dim=-1)
  
  if topk is not None:
    top_probs = torch.topk(probs, topk)
    mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
    mask = mask.sum(dim=1)
    probs *= mask
    probs /= probs.sum(dim=-1)
  
  if nucleus != 1:
    probs_sorted = torch.sort(probs, descending=True, dim=-1)
    sorted_indices = probs_sorted.indices
    sorted_values = probs_sorted.values

    cumsum = torch.cumsum(sorted_values, dim=-1)
    ks = (cumsum < nucleus).long().sum(dim=-1)
    ks = torch.max(ks, torch.ones_like(ks))

    # TODO: Make this more efficient using gather
    ks = F.one_hot(ks, probs.shape[-1]).float()
    cutoffs = (sorted_values * ks).sum(-1)

    mask = (probs > cutoffs.unsqueeze(1)).float()
    probs *= mask
    
    probs /= probs.sum(keepdim=True, dim=-1)

  next_tokens = torch.multinomial(probs, num_samples=1)

  return next_tokens

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
#     print(y_tensor, y.size(), y_tensor.device)
    zeros = torch.ones(*y.size(), num_classes, dtype=y.dtype, device= y_tensor.device)
    zscatter =  zeros.scatter(scatter_dim, y_tensor, 0)
    #print(zscatter)
    zscatter_cons =  torch.min(zscatter, dim=1)[0]
    
    return zscatter_cons

def _to_one_hot_fill(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
#     print(y_tensor, y.size())
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype, device= y_tensor.device)
    zscatter =  zeros.scatter(scatter_dim, y_tensor, 1)
#     print(zscatter)
    zscatter_cons =  torch.max(zscatter, dim=1)[0]
    
    return zscatter_cons

def infill_with_ilm(
  model,
  special_tokens_to_ids,
  x,
  num_infills=1,
  max_sequence_length=256,
  nucleus=0.95):
  
  _sep_id = special_tokens_to_ids['<|startofinfill|>']
  _end_span_id = special_tokens_to_ids['<|endofinfill|>']
  _special_ids = special_tokens_to_ids.values()
  
  # Make sure example doesn't already ends with [sep]
  if x[-1] == _sep_id:
    x = x[:-1]
  
  # Count number of blanks
  blank_idxs = []
  for i, tok_id in enumerate(x):
    if tok_id in _special_ids:
      blank_idxs.append(i)
  k = len(blank_idxs)
  if k == 0:
    raise ValueError()
  
  # Decode until we have that many blanks
  with torch.no_grad():
    device = next(model.parameters()).device
    context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
    
    terminated = []

    while context.shape[0] > 0:
      logits = model(context)[0][:, -1]
      next_tokens = sample_from_logits(logits, nucleus=nucleus)
      context = torch.cat((context, next_tokens), dim=1)
      
      num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
      # print(num_predicted_spans)
      # print(context.size())
#       import pdb;pdb.set_trace()
      terminate_expected = num_predicted_spans >= k
      terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
      terminate = terminate_expected | terminate_toolong
      
      if torch.any(terminate):
        terminated_seqs = context[terminate, len(x)+1:]
        terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
        context = context[~terminate, :]
  
  # Collect generated spans
  generated_spans = []
  for gen in terminated:
    spans = []
    while _end_span_id in gen:
      spans.append(gen[:gen.index(_end_span_id)])
      gen = gen[gen.index(_end_span_id) + 1:]
    while len(spans) < k:
      spans.append([])
    generated_spans.append(spans)
  
  # Insert into context
  generated = []
  for spans in generated_spans:
    context = copy.deepcopy(x)
    for i, j in enumerate(blank_idxs[::-1]):
      del context[j]
      context[j:j] = spans[k - 1 - i]
    generated.append(context)

  return generated


def sample_from_logits_withexc(
    logits,
    avoid_ids,
    avoid_ids_general=None,
    promote_ids_general=None,
    temp=1.,
    topk=None,
    nucleus=1.):
  if temp == 0:
    return torch.argmax(logits, dim=-1).unsqueeze(-1)
  elif temp != 1:
    logits /= temp
  
  probs = F.softmax(logits, dim=-1)
  # import pdb;pdb.set_trace()
  if avoid_ids is not None and avoid_ids.shape[1]>0:
      avoid_probs = _to_one_hot(avoid_ids, logits.size()[1])
      avoid_probs_mask = avoid_probs
#       print(avoid_probs.size(), logits.size())
  if avoid_ids_general is not None:
      avoid_ids_general_vals = _to_one_hot_fill(avoid_ids_general, logits.size()[1])
      avoid_ids_general_vals = avoid_ids_general_vals.float()*0.8
      avoid_ids_general_vals = 1- avoid_ids_general_vals
  if promote_ids_general is not None:
      promote_ids_general_vals = _to_one_hot_fill(promote_ids_general, logits.size()[1])
      promote_ids_general_vals = avoid_ids_general_vals.float()*20.0
#   print(avoid_ids_general_vals, avoid_ids_general_vals.size(), 'avoid_ids_general_vals')
  if topk is not None:
    top_probs = torch.topk(probs, topk)
    mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
    mask = mask.sum(dim=1)
    probs *= mask
    # probs /= probs.sum(dim=-1)
  
  if nucleus != 1:
        
    if avoid_ids is not None and avoid_ids.shape[1]>0:
#         print(avoid_probs_mask.size(),avoid_ids_general_mask.size())
        # import pdb;pdb.set_trace()
        probsn = probs * (avoid_probs_mask)
        # print((probs ).sum(keepdim=True, dim=-1), 'org mask sum')
        # print((probs * (1-avoid_probs_mask)).sum(keepdim=True, dim=-1), 'rev mask sum')
        # print(probsn, ' apply mask')
        # print('-----------',  (probsn.sum(keepdim=True, dim=-1) == 0).nonzero() )
#         print('in000 ', avoid_probs_mask.size(), probs.size(), probsn.sum(keepdim=True, dim=-1))
        if avoid_ids_general is not None:
            probs *= avoid_ids_general_vals
        if promote_ids_general is not None:
#             print(probs)
            probs *= promote_ids_general_vals
#             print(probs, promote_ids_general_vals)
            
        probsn /= probsn.sum(keepdim=True, dim=-1)
        probs = probsn 
#         print(probs, 'probs', probs.size())
#         print(probsn, ' after mask')
#     else:
#         probs /= probs.sum(keepdim=True, dim=-1)
    probs_sorted = torch.sort(probs, descending=True, dim=-1)
    sorted_indices = probs_sorted.indices
    sorted_values = probs_sorted.values

    cumsum = torch.cumsum(sorted_values, dim=-1)
    ks = (cumsum < nucleus).long().sum(dim=-1)
    ks = torch.max(ks, torch.ones_like(ks))

    # TODO: Make this more efficient using gather
    ks = F.one_hot(ks, probs.shape[-1]).float()
    cutoffs = (sorted_values * ks).sum(-1)

    mask = (probs > cutoffs.unsqueeze(1)).float()
    probs *= mask
    probs /= probs.sum(keepdim=True, dim=-1)

#     print('000 ', probs.size(), probs.sum(keepdim=True, dim=-1))
   
#   print('check isnan', torch.isnan(probs))

  next_tokens = torch.multinomial(probs, num_samples=1)
  # print(next_tokens)
  return next_tokens



def infill_function_withexp(
  model,
  special_tokens_to_ids,
  x,
  avoid_ids,
  avoid_ids_general=None,
  promote_ids_general=None,
  num_infills=1,
  max_sequence_length=256,
  nucleus=0.95):

  '''
  avoid_ids : These tokens should not be generated they are a list of list of tokens for each blank
  avoid_ids_general: These tokens are demoted with p porbablity, for example tokens related to blanked tokens. 
    Currently they are selected for all blanks and are not per blank basis
  promote_ids_general: These tokens are promoted with p porbablity
  '''
  
  _sep_id = special_tokens_to_ids['<|startofinfill|>']
  _end_span_id = special_tokens_to_ids['<|endofinfill|>']
  _special_ids = special_tokens_to_ids.values()
  
  # Make sure example doesn't already ends with [sep]
  if x[-1] == _sep_id:
    x = x[:-1]
  
  # Count number of blanks
  blank_idxs = []
  for i, tok_id in enumerate(x):
    if tok_id in _special_ids:
      blank_idxs.append(i)
  k = len(blank_idxs)
  if k == 0:
    raise ValueError()

  maxlen = max([len(x) for x in avoid_ids])
  avoid_ids = [x + [0]*(maxlen-len(x)) for x in avoid_ids]
  avoid_ids = [avoid_ids]*num_infills

  # Decode until we have that many blanks
  with torch.no_grad():
    device = next(model.parameters()).device
    context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
    if avoid_ids_general is not None:
        avoid_ids_general = torch.tensor(avoid_ids_general, dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
    if promote_ids_general is not None:
        promote_ids_general = torch.tensor(promote_ids_general, dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
#     print(avoid_ids_general)
    avoid_ids = torch.tensor(avoid_ids, dtype=torch.long, device=device)
    terminated = []
    next_avoid = torch.index_select(avoid_ids, 1, torch.zeros(num_infills, dtype=avoid_ids.dtype, device=device))[0]
    coun = 0
    while context.shape[0] > 0:
      logits = model(context)[0][:, -1]
      avoid_ids_general_to_use = None
      if avoid_ids_general is not None:
          avoid_ids_general_to_use = avoid_ids_general[:logits.size(0),:]
      promote_ids_general_to_use = None
      if promote_ids_general is not None:
          promote_ids_general_to_use = promote_ids_general[:logits.size(0),:]
      # print(next_avoid)
      # import pdb;pdb.set_trace()
      next_tokens = sample_from_logits_withexc(logits, next_avoid, avoid_ids_general=avoid_ids_general_to_use, promote_ids_general=promote_ids_general_to_use, nucleus=nucleus)
      context = torch.cat((context, next_tokens), dim=1)
      
      num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
#       print('now', context.size(), avoid_ids.size())
#       print('num_predicted_spans', num_predicted_spans)
#       print(avoid_ids)
      index_predicted_spans = torch.clamp(num_predicted_spans, max=k-1)
#       print(index_predicted_spans)
#       print('avoid now at',avoid_ids.size(), num_predicted_spans.size() )
      next_avoid = torch.index_select(avoid_ids, 1, index_predicted_spans)[0]
#       print(next_avoid)
#       print(avoid_ids[:,num_predicted_spans])
#       import pdb;pdb.set_trace()
      terminate_expected = num_predicted_spans >= k
      terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
      terminate = terminate_expected | terminate_toolong
      
      if torch.any(terminate):
        terminated_seqs = context[terminate, len(x)+1:]
        terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
        context = context[~terminate, :]
        avoid_ids = avoid_ids[~terminate, :]
        next_avoid = next_avoid[~terminate, :]
#         print('ter', context.size(), avoid_ids.size())
  
  # Collect generated spans
  generated_spans = []
  for gen in terminated:
    spans = []
    while _end_span_id in gen:
      spans.append(gen[:gen.index(_end_span_id)])
      gen = gen[gen.index(_end_span_id) + 1:]
    while len(spans) < k:
      spans.append([])
    generated_spans.append(spans)
  
  # Insert into context
  generated = []
  for spans in generated_spans:
    context = copy.deepcopy(x)
    for i, j in enumerate(blank_idxs[::-1]):
      del context[j]
      context[j:j] = spans[k - 1 - i]
    generated.append(context)

  return generated


def infill_naive_with_ilm(
  model,
  special_tokens_to_ids,
  x,
  num_infills=1,
  max_sequence_length=256,
  nucleus=0.95):
  
  _sep_id = special_tokens_to_ids['<|startofinfill|>']
  _end_span_id = special_tokens_to_ids['<|endofinfill|>']
  _special_ids = special_tokens_to_ids.values()
  
  # Make sure example doesn't already ends with [sep]
  if x[-1] == _sep_id:
    x = x[:-1]
  
  # Count number of blanks
  # blank_idxs = []
  # for i, tok_id in enumerate(x):
  #   if tok_id in _special_ids:
  #     blank_idxs.append(i)
  # k = len(blank_idxs)
  # if k == 0:
  #   raise ValueError()
  
  # Decode until we have that many blanks
  with torch.no_grad():
    device = next(model.parameters()).device
    context = torch.tensor(x + [_sep_id], dtype=torch.long, device=device).unsqueeze(0).repeat(num_infills, 1)
    
    terminated = []

    while context.shape[0] > 0:
      logits = model(context)[0][:, -1]
      next_tokens = sample_from_logits(logits, nucleus=nucleus)
      context = torch.cat((context, next_tokens), dim=1)
      
      num_predicted_spans = (context == _end_span_id).long().sum(dim=1)
      
      terminate_expected = num_predicted_spans >= 1
      terminate_toolong = torch.ones_like(context).long().sum(dim=1) >= max_sequence_length
      terminate = terminate_expected | terminate_toolong
      
      if torch.any(terminate):
        terminated_seqs = context[terminate, len(x)+1:]
        terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
        context = context[~terminate, :]
  
  # Collect generated spans
  # print(terminated)
  # generated_spans = []
  # for gen in terminated:
  #   spans = []
  #   while _end_span_id in gen:
  #     spans.append(gen[:gen.index(_end_span_id)])
  #     gen = gen[gen.index(_end_span_id) + 1:]
  #   while len(spans) < k:
  #     spans.append([])
  #   generated_spans.append(spans)
  
  # Insert into context
  # generated = []
  # for spans in generated_spans:
  #   context = copy.deepcopy(x)
  #   for i, j in enumerate(blank_idxs[::-1]):
  #     del context[j]
  #     context[j:j] = spans[k - 1 - i]
  #   generated.append(context)

  generated_spans = []
  for gen in terminated:
    spans = []
    while _end_span_id in gen:
      spans.append(gen[:gen.index(_end_span_id)])
      gen = gen[gen.index(_end_span_id) + 1:]
    while len(spans) < 1:
      spans.append([])
    generated_spans.append(spans)

  generated = []
  # for spans in generated_spans:
  context = copy.deepcopy(x)
  # for i, j in enumerate(blank_idxs[::-1]):
  #   del context[j]
  #   context[j:j] = spans[k - 1 - i]
  # if len(spans)==0: spans=[10]
  generated.append(context+ [_sep_id] +spans[0])

  return generated
