import sys, os
from tqdm import tqdm

import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    print('Syntax: {} <in_file.csv> <out_file.csv> <directory>'.format(sys.argv[0]))
    sys.exit(0)

(in_file, out_file, directory) = sys.argv[1:]

in_df = pd.read_csv(in_file)
out_df = pd.DataFrame(index = np.arange(len(in_df)), columns = list(in_df))
print(in_df)
print(out_df)


for i, row in tqdm(in_df.iterrows(), total = len(in_df)):
    if os.path.exists(directory + row['id'] + '.jpg'):
        out_df.loc[i] = row

out_df = out_df.dropna(how = 'any')

print(out_df)
out_df.to_csv(out_file, index = False)
