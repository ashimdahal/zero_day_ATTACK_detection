import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

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

label_mapping = {
    # Type: probe
    "back.": 1,
    "land.": 1,
    "neptune.": 1,
    "pod.": 1,
    "smurf.": 1,
    "teardrop.": 1,
    "apache2.": 1,
    "udpstorm.": 1,
    "processtable.": 1,
    "worm.": 1,
    # Type: DOS
    "satan.": 2,
    "ipsweep.": 2,
    "nmap.": 2,
    "portsweep.": 2,
    "mscan.": 2,
    "saint.": 2,
    # Type: Unauthorized Access
    "guess_passwd.": 3,
    "ftp_write.": 3,
    "imap.": 3,
    "phf.": 3,
    "multihop.": 3,
    "warezmaster.": 3,
    "warezclient.": 3,
    "spy.": 3,
    "xlock.": 3,
    "xsnoop.": 3,
    "snmpguess.": 3,
    "snmpgetattack.": 3,
    "httptunnel.": 3,
    "sendmail.": 3,
    "named.": 3,
    "mailbomb.": 3,
    "buffer_overflow.": 3,
    "loadmodule.": 3,
    "rootkit.": 3,
    "perl.": 3,
    "sqlattack.": 3,
    "xterm.": 3,
    "ps.": 3,
    # Type: Normal
    "normal.": 0,
}

df.replace(label_mapping, inplace=True)


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
y = torch.tensor(df["label"].values, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=42
)

# making a datset
train_ds, valid_ds = TensorDataset(X_train, y_train), TensorDataset(  # type:ignore
    X_test, y_test  # type:ignore
)

# Make a dataloader

BATCH = 1024
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH, shuffle=True)


def accuracy(label, preds):
    preds = torch.argmax(preds, dim=1)
    accuracy = (label == preds).sum() / len(preds)

    return accuracy


class BaseNet(nn.Module):
    def get_loss(self, batch, loss_fn):
        features, labels = batch
        preds = self(features)
        loss = loss_fn(preds, labels)
        return loss

    def validate(self, batch, loss_fn):
        feature, labels = batch
        loss = self.get_loss(batch, loss_fn)
        pred = self(feature)

        acc = accuracy(labels, pred)
        return {"valid_loss": loss, "valid_acc": acc}

    def average_validation(self, out):
        loss = torch.stack([loss["valid_loss"] for loss in out]).mean()
        acc = torch.stack([loss["valid_acc"] for loss in out]).mean()
        return {"valid_loss": loss.item(), "valid_acc": acc.item()}

    def log_epoch(self, e, epoch, res):
        print(
            "[{} / {}] epoch/s, training loss is {:.4f} "
            "validation loss is {:.4f}, validation accuracy "
            "is {:.4f} ".format(
                e + 1, epoch, res["train_loss"], res["valid_loss"], res["valid_acc"]
            )
        )


class ClassifierMLP(BaseNet):
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
