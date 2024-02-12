from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


df: pd.DataFrame = pd.read_csv("./df_with_predictions.tar")
cols = list(df.columns)

X, y = df.remove("label", axis=1).values, df["label"].values
