import torch
import numpy as np
import pandas as pd
import ast
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(
    "data/dontpatronizeme_pcl.tsv",
    sep="\t",
    skiprows=4,
    header=None
)

data.columns = [
    "id",
    "par_id",
    "keyword",
    "country",
    "paragraph",
    "label_raw"
]

data["id"] = data["id"].astype(str)
data["paragraph"] = data["paragraph"].fillna("").astype(str)

dev_labels = pd.read_csv("data/dev_semeval_parids-labels.csv")
dev_labels["par_id"] = dev_labels["par_id"].astype(str)

dev_df = data.merge(
    dev_labels[["par_id"]],
    left_on="id",
    right_on="par_id"
)

dev_df = dev_df[["paragraph"]].reset_index(drop=True)

test_df = pd.read_csv(
    "data/task4_test.tsv",
    sep="\t",
    header=None
)

test_df.columns = [
    "id",
    "par_id",
    "keyword",
    "country",
    "paragraph"
]

test_df["paragraph"] = test_df["paragraph"].fillna("").astype(str)
test_df = test_df[["paragraph"]].reset_index(drop=True)

print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("BestModel").to(device)
model.eval()

def predict(df):
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["paragraph"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)

dev_probs = predict(dev_df)
test_probs = predict(test_df)

dev_labels_full = pd.read_csv("data/dev_semeval_parids-labels.csv")
dev_labels_full["label"] = dev_labels_full["label"].apply(lambda x: 1 if sum(ast.literal_eval(x)) > 0 else 0)
true_dev = dev_labels_full["label"].values

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (dev_probs >= t).astype(int)
    f1 = f1_score(true_dev, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Dev Threshold:", best_threshold)
print("Best Dev F1:", best_f1)

dev_preds = (dev_probs >= best_threshold).astype(int)
test_preds = (test_probs >= best_threshold).astype(int)

np.savetxt("dev.txt", dev_preds, fmt="%d")
np.savetxt("test.txt", test_preds, fmt="%d")

print("Saved dev.txt and test.txt")