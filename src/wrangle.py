import pandas as pd
from lib import standardize_text

text_col = "Descripción de la mercancía"
label_col = "Tarifario"
df = pd.read_parquet("data/top1k.parquet")

# Dedup
df.loc[:, text_col] = df[text_col].apply(lambda x: standardize_text(x))
df = df.drop_duplicates(text_col)

# Balance
sample_count = df.groupby(label_col).count()
under_sampled = int(sample_count[text_col].min())
under_sampled

# Sampling
# splitter = GroupShuffleSplit(test_size=0.20, n_splits=2, random_state=7)
# split = splitter.split(df, groups=df["Group_Id"])
# train_inds, test_inds = next(split)
#
# train = df.iloc[train_inds]
# test = df.iloc[test_inds]

grouped_df = df.groupby(label_col).sample(n=15, random_state=42)  # suffle group

# split
train = grouped_df.groupby(label_col).head(10)
test = grouped_df.groupby(label_col).tail(5)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
