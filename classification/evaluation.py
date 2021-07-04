from IPython import embed
import pytrec_eval

METRICS = {'map',
           'recip_rank',
           'ndcg_cut',
           'recall'}

RECALL_AT_W_CAND = {
                    'R_1@1',
                    'R_2@2',
                    'R_2@1',
                    'R_3@1',
                    'R_4@1',
                    'R_5@1',
                    'R_6@1',
                    'R_6@2', 'R_6@3', 'R_6@3', 'R_6@4', 'R_6@5',
                    'R_10@1',
                    # 'R_33@1'
                    # 'R_10@5',
                    }

def recall_at_with_k_candidates(preds, labels, k, at):
    """
    Calculates recall with k candidates. labels list must be sorted by relevance.

    Args:
        preds: float list containing the predictions.
        labels: float list containing the relevance labels.
        k: number of candidates to consider.
        at: threshold to cut the list.
        
    Returns: float containing Recall_k@at
    """
    num_rel = labels.count(1)
    #'removing' candidates (relevant has to be in first positions in labels)
    preds = preds[:k]
    labels = labels[:k]

    sorted_labels = [x for _,x in sorted(zip(preds, labels), reverse=True)]
    hits = sorted_labels[:at].count(1)
    return hits/num_rel

def calculate_classification_metrics(preds, label_ids):
    preds = [1 if p >= 0.5 else 0 for p in preds]
    tp = [1 if pred == label == 1 else 0 for pred, label in zip(preds, label_ids)]
    tn = [1 if pred == label == 0 else 0 for pred, label in zip(preds, label_ids)]
    fp = [1 if pred == 1 and label == 0 else 0 for pred, label in zip(preds, label_ids)]
    fn = [1 if pred == 0 and label == 1 else 0 for pred, label in zip(preds, label_ids)]
    acc = [1 if pred == label else 0 for pred, label in zip(preds, label_ids)]
    return {"accuracy": sum(acc),
            'tp': sum(tp),  # labels are not flipped
            'tn': sum(tn),
            'fp': sum(fp),
            'fn': sum(fn),
            "avg_accuracy": sum(acc)/float(len(preds)),
            'avg_tp': sum(tp)/float(sum(tp)+ sum(fn)),  # labels are not flipped
            'avg_tn': sum(tn)/float(sum(fp)+ sum(tn)),
            'avg_fp': sum(fp)/float(sum(fp)+ sum(tn)),
            'avg_fn': sum(fn)/float(sum(tp)+ sum(fn))
            }

def evaluate_classification_metrics(results):
    for model in results.keys():
        preds_r = results[model]['preds']
        labels_r = results[model]['labels']
        softmax_l1_r = results[model]['l1_pred']

        preds = []
        labels = []
        for i, p in enumerate(softmax_l1_r):
            for j, _ in enumerate(range(len(p))):
                labels.append(labels_r[i][j])
                preds.append(softmax_l1_r[i][j])

        classification_dict = calculate_classification_metrics(preds, labels)
        results[model]['eval'] = {}
        results[model]['eval']['classification_results'] = classification_dict

    return results


def evaluate_models(results):
    """
    Calculate METRICS for each model in the results dict
    
    Args:
        results: dict containing one key for each model and inside them pred and label keys. 
        For example:    
             results = {
              'model_1': {
                 'preds': [[1,2],[1,2]],
                 'labels': [[1,2],[1,2]]
               }
            }.
    Returns: dict with the METRIC results per model and query.
    """    
    for model in results.keys():
        preds = results[model]['preds']
        labels = results[model]['labels']
        softmax_l1 = results[model]['l1_pred']
        run = {}
        qrel = {}
        runst = {}
        for i, p in enumerate(preds):
            run['q{}'.format(i+1)] = {}
            qrel['q{}'.format(i+1)] = {}
            runst['q{}'.format(i + 1)] = {}
            for j, _ in enumerate(range(len(p))):
                run['q{}'.format(i+1)]['d{}'.format(j+1)] = float(preds[i][j])
                qrel['q{}'.format(i + 1)]['d{}'.format(j + 1)] = int(labels[i][j])
                runst['q{}'.format(i + 1)]['d{}'.format(j + 1)] = float(softmax_l1[i][j])

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, METRICS)
        # results[model]['eval'] = evaluator.evaluate(run)
        results[model]['eval'] = evaluator.evaluate(runst)

        # print(results[model]['eval'])
        # print(evaluator.evaluate(runst))

        for query in qrel.keys(): 
            preds = []
            labels = []
            for doc in run[query].keys():
                preds.append(run[query][doc])
                labels.append(qrel[query][doc])
            
            for recall_metric in RECALL_AT_W_CAND:
                cand = int(recall_metric.split("@")[0].split("R_")[1])
                at = int(recall_metric.split("@")[-1])
                results[model]['eval'][query][recall_metric] = recall_at_with_k_candidates(preds, labels, cand, at)

    return results
