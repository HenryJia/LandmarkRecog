import pandas as pd
train_full = pd.read_csv("train.csv")
train_sample = train_full.sample(n = int(1e5))
train_sample.to_csv("train_sample.csv", index = False)

