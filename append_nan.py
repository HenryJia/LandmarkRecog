import sys, os
import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    print('Syntax: {} <test.csv> <submission.csv> <submission_out.csv>'.format(sys.argv[0]))
    sys.exit(0)

(test, submission, submission_out) = sys.argv[1:]

test_df = pd.read_csv(test)
in_df = pd.read_csv(submission)
print(test_df)
print(in_df)

out_df = in_df
idxs = list(in_df['id'])
for i, row in tqdm(test_df.iterrows(), total = len(test_df)):
    if row['id'] not in idxs:
        out_df = out_df.append({'id' : row['id'], 'landmarks' : ' '}, ignore_index = True)


print(out_df)
out_df.to_csv(submission_out, index = False)
