import sys, os
from tqdm import tqdm

import numpy as np
import pandas as pd

from train_utils import split_validation

if len(sys.argv) != 4:
    print('Syntax: {} <in_file.csv> <out_train.csv> <out_val.csv>'.format(sys.argv[0]))
    sys.exit(0)

(in_file, out_train, out_val) = sys.argv[1:]

in_df = pd.read_csv(in_file)
train_data, val_data = split_validation(in_df, 0.8)

print('Training samples: ', len(train_data), '\n', train_data.head())
print('Validation samples: ', len(val_data), '\n', val_data.head())

train_data.to_csv(out_train, index = False)
val_data.to_csv(out_val, index = False)
