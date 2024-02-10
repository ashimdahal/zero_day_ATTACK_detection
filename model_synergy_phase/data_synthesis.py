import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np


col_names = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

df: pd.DataFrame = pd.read_csv("../data/kddcup.data.gz", names=col_names)


num_cols = df._get_numeric_data().columns
cate_cols = list(set(df.columns) - set(num_cols))

category_mapping = {}
for category in cate_cols:
    labels, unique_values = pd.factorize(df[category])
    mapping = {value: label for label, value in enumerate(unique_values)}
    df[category] = labels
    category_mapping[category] = mapping

df: pd.DataFrame = df[[col for col in df if df[col].nunique() > 1]]  # type:ignore

X = torch.tensor(df.drop("label", axis=1).values.astype(np.float32))

synthesized_dl = DataLoader(TensorDataset(X), batch_size=len(X), shuffle=False)


class ClassifierMLP(nn.Module):
    def __init__(self, activation, input_dim, hidden_1, hidden_2, out):
        super().__init__()
        self.hidden_1 = nn.Linear(input_dim, hidden_1)
        self.hidden_2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, out)
        self.activation = activation

    def forward(self, x):
        x1 = self.hidden_1(x)
        x2 = self.activation(x1)
        x3 = self.hidden_2(x2)
        x4 = self.activation(x3)
        out = self.out(x4)

        return out


activation = nn.ReLU()
input_dim = X.shape[1]
hidden_1, hidden_2 = 128, 64
out = df["label"].nunique()

truncated_model = ClassifierMLP(activation, input_dim, hidden_1, hidden_2, out)
weighted_truncated_model = ClassifierMLP(activation, input_dim, hidden_1, hidden_2, out)

truncated_model.load_state_dict(torch.load("../models/model_truncated_four_class.pth"))
weighted_truncated_model.load_state_dict(
    torch.load("../models/model_truncated_weighted_four_class.pth")
)

truncated_model.eval()
weighted_truncated_model.eval()

# build a machine learning layer here and use it to test the given models
# build a machine learning layer here and use it to test the given models

with torch.no_grad():
    for data in synthesized_dl:
        output_truncated = truncated_model(data)
        output_weighted_truncated = weighted_truncated_model(data)

        flat_truncated = output_truncated.view(-1).to("cpu").numpy()
        flat_weighted_truncated = output_weighted_truncated.view(-1).to("cpu").numpy()

        df["truncated_predictions"] = flat_truncated
        df["weighted_truncated_predictions"] = flat_truncated

df.to_csv("./df_with_predictions.csv")
