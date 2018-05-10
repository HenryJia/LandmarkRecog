import os, sys
import math

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''

    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap


if len(sys.argv) != 3:
    print('Syntax: {} <train.csv> <submission.csv>'.format(sys.argv[0]))
    sys.exit(0)

(train_csv, submission_csv) = sys.argv[1:]
train_data = pd.read_csv(train_csv)
submission_data = pd.read_csv(submission_csv)
train_data = train_data.set_index('id')
print(train_data)

categories = []
confidence = []
targets = []
for i, row in tqdm(submission_data.iterrows(), total = len(submission_data)):
    cat, conf = row['landmarks'].split(' ')
    if len(cat) > 0:
        categories += [int(cat)]
        confidence += [float(conf)]
        targets += [int(train_data['landmark_id'].loc[row['id']])]

print(len(categories), len(confidence), len(targets))

print(GAP_vector(np.array(categories).flatten(), np.array(confidence).flatten(), np.array(targets).flatten()))
