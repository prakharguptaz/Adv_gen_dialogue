import evaluation
import numpy as np
import torch
from typing import List, Tuple
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction)

def compute_metrics(p: EvalPrediction, is_regression=False):
    preds_logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    
    #labels were flipped during data creation, so accomodating with legacy choice for both predictions and labels
    preds_logits[:, [1, 0]] = preds_logits[:, [0, 1]] 
    label_ids = p.label_ids
    label_ids =  [1 if x==0 else 0 for x in label_ids]
    preds = np.argmax(preds_logits, axis=1)
    softmax_l1 = softmax(preds_logits, axis=1)[:, 1].tolist()

    tp = [1 if pred == label ==1 else 0 for pred,label in zip(preds, label_ids)]
    tn = [1 if pred == label ==0 else 0 for pred,label in zip(preds, label_ids)]
    fp = [1 if pred == 1 and label ==0 else 0 for pred,label in zip(preds, label_ids)]
    fn = [1 if pred == 0 and label ==1 else 0 for pred,label in zip(preds, label_ids)]
    classification_metric = {"accuracy": (preds == label_ids).astype(np.float32).mean().item(),
            'tp': sum(tp), #labels are flipped
            'tn': sum(tn),
            'fp': sum(fp),
            'fn': sum(fn)
            }

    metrics = dict()
    metrics.update(classification_metric)
    agg_results = evaluate_and_aggregate(preds.tolist(), label_ids, softmax_l1, ['R_1@1', 'R_2@1','R_3@1','R_4@1','R_5@1', 'R_6@1', 'R_6@2', 'R_6@3', 'R_6@3', 'R_6@4', 'R_6@5', 'R_10@1', 'R_33@1', 'R_2@2', 'ndcg_cut_5', 'map'])
    # print(agg_results)
    metrics.update(agg_results)

    return metrics

def evaluate(preds, labels, softmax_l1):
    qrels = {}
    qrels['model'] = {}
    qrels['model']['preds'] = preds
    qrels['model']['labels'] = labels
    qrels['model']['l1_pred'] = softmax_l1
    results = evaluation.evaluate_models(qrels)
    return results

def evaluate_classification(preds, labels, softmax_l1):
    qrels = {}
    qrels['model'] = {}
    qrels['model']['preds'] = preds
    qrels['model']['labels'] = labels
    qrels['model']['l1_pred'] = softmax_l1

    results = evaluation.evaluate_classification_metrics(qrels)
    return results

    
def get_mrr(labels_ids, softmax_l1):
    total_rank = []
    for i, labels in enumerate(labels_ids):
        softmaxs = softmax_l1[i]
        seq = sorted(softmaxs,reverse=True)
        index = [seq.index(v)+1 for v in softmaxs]
        #print(index)
        for j,l in enumerate(labels):
            if labels[j]==1:
                total_rank.append(1/index[j])
    avg_mrr = sum(total_rank)/len(total_rank)
    
    return avg_mrr


def evaluate_and_aggregate(preds, labels, softmax_l1, metrics):
    """
    Calculate evaluation metrics for a pair of preds and labels.
    
    Aggregates the results only for the evaluation metrics in metrics arg.

    Args:
        preds: list of lists of floats with predictions for each query.
        labels: list of lists with of floats with relevance labels for each query.
        softmax_l1: value for label 1 after softmax over both labels for each query.
        metrics: list of str with the metrics names to aggregate.
        
    Returns: dict with the METRIC results per model and query.
    """
    #acumulate_list_multiple_relevant - ensure that you have only one 1 followed by n zeros for ranking involving single positive
    labels = acumulate_list_multiple_relevant(labels)
    preds = acumulate_l1_by_l2(preds, labels)
    softmax_l1 = acumulate_l1_by_l2(softmax_l1, labels)
    results = evaluate(preds, labels, softmax_l1)

    classification_results = evaluate_classification(preds, labels, softmax_l1)


    agg_results = {}
    for metric in metrics:
        res = 0
        per_q_values = []
        for q in results['model']['eval'].keys():
            per_q_values.append(results['model']['eval'][q][metric])
            res += results['model']['eval'][q][metric]
        res /= len(results['model']['eval'].keys())
        agg_results[metric] = res

    for classification_metric in classification_results['model']['eval']['classification_results']:
        agg_results[classification_metric] = classification_results['model']['eval']['classification_results'][classification_metric]

    avg_mrr = get_mrr(labels, softmax_l1)
    agg_results['mrr'] = avg_mrr

    return agg_results


def acumulate_list(l : List[float], acum_step: int) -> List[List[float]]:
    """
    Splits a list every acum_step and generates a resulting matrix

    Args:
        l: List of floats

    Returns: List of list of floats divided every acum_step

    """
    acum_l = []    
    current_l = []    
    for i in range(len(l)):
        current_l.append(l[i])        
        if (i + 1) % acum_step == 0 and i != 0:
            acum_l.append(current_l)
            current_l = []
    return acum_l


def acumulate_list_multiple_relevant(l : List[float]) -> List[List[float]]:
    """
    Splits a list that has variable number of labels 1 first followed by N 0.

    Args:
        l: List of floats

    Returns: List of list of floats divided every set of 1s followed by 0s.
    Example: [1,1,1,0,0,1,0,0,1,1,0,0] --> [[1,1,1,0,0], [1,0,0], [1,1,0,0]]

    """    
    acum_l = []
    current_l = []
    for i in range(len(l)):
        current_l.append(l[i])
        if (i == len(l)-1) or (l[i] == 0 and l[i+1] == 1):
            acum_l.append(current_l)
            current_l = []
    return acum_l

def acumulate_l1_by_l2(l1 : List[float], l2 : List[List[float]]) -> List[List[float]]:
    """
    Splits a list (l1) using the shape of l2.

    Args:
        l: List of floats

    Returns: List of list of floats divided by the shape of l2
    Example: [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5] , [[1,1,1,0,0], [1,0,0], [1,1,0,0]]--> 
             [[0.5,0.5,0.5,0.5,0.5], [0.5,0.5,0.5], [0.5,0.5,0.5,0.5]]

    """    
    acum_l1 = []
    l1_idx = 0
    for l in l2:
        current_l = []
        for i in range(len(l)):
            current_l.append(l1[l1_idx])
            l1_idx+=1
        acum_l1.append(current_l)
    return acum_l1


def from_df_to_list_without_nans(df) -> List[float]:
    res = []
    for r in df.itertuples(index=False):
        l = []
        for c in range(df.shape[1]):
            if str(r[c]) != 'nan':
                l.append(r[c])
        res.append(l)
    return res
