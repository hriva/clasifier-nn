import pandas as pd
from lib import standardize_text

text_col = "Descripción de la mercancía"
label_col = "Tarifario"
df = pd.read_parquet("data/top1k.parquet")
df = df.loc[df[label_col] != "2711120100"]

# Dedup
df.loc[:, text_col] = df[text_col].apply(lambda x: standardize_text(x))
df = df.drop_duplicates(text_col)

# Balance
sample_count = df.groupby(label_col).count()
under_sampled = int(sample_count[text_col].min())
under_sampled
grouped_df = df.groupby(label_col).sample(n=50, random_state=42)  # suffle group
grouped_df.reset_index(drop=True, inplace=True)

# dump
grouped_df.to_parquet("data/data.parquet")
