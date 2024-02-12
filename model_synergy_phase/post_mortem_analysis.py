import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

import torch

# from torch.utils.data import TensorDataset, DataLoader

df: pd.DataFrame = pd.read_csv("./df_with_predictions.tar")
cols = list(df.columns)

X, y = torch.tensor(df.drop("label", axis=1).values), torch.tensor(df["label"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=42
)


clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

joblib.dump(clf, "randomforrest_classifier.joblib")
