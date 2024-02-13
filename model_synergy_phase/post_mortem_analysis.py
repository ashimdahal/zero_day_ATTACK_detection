import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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


rf_clf = RandomForestClassifier(max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)
joblib.dump(rf_clf, "randomforrest_classifier.joblib")


dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_clf.fit(X, y)
joblib.dump(dt_clf, "decisiontree_classifier.joblib")

svc_clf = LinearSVC(dual="auto", tol=1e-05, random_state=42)
svc_clf.fit(X, y)
joblib.dump(svc_clf, "supportvector_classifier.joblib")

lr_clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=3000)
lr_clf.fit(X, y)
joblib.dump(lr_clf, "LogisticRegression_classifier.joblib")
