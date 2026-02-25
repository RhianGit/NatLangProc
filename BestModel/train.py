import torch
import numpy as np
import pandas as pd
import ast
from torch import nn
from sklearn.metrics import f1_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv(
    "dontpatronizeme_pcl.tsv",
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

train_labels = pd.read_csv("train_semeval_parids-labels.csv")
dev_labels = pd.read_csv("dev_semeval_parids-labels.csv")

def convert_to_binary(label_string):
    label_list = ast.literal_eval(label_string)
    return 1 if sum(label_list) > 0 else 0

train_labels["label"] = train_labels["label"].apply(convert_to_binary)
dev_labels["label"] = dev_labels["label"].apply(convert_to_binary)

data["id"] = data["id"].astype(str)
train_labels["par_id"] = train_labels["par_id"].astype(str)
dev_labels["par_id"] = dev_labels["par_id"].astype(str)

train_df = data.merge(train_labels[["par_id", "label"]],
                      left_on="id",
                      right_on="par_id")

dev_df = data.merge(dev_labels[["par_id", "label"]],
                    left_on="id",
                    right_on="par_id")

train_df = train_df[["paragraph", "label"]]
dev_df = dev_df[["paragraph", "label"]]

train_df["paragraph"] = train_df["paragraph"].fillna("").astype(str)
dev_df["paragraph"] = dev_df["paragraph"].fillna("").astype(str)

print("Train size:", len(train_df))
print("Dev size:", len(dev_df))

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(
        batch["paragraph"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
dev_dataset = Dataset.from_pandas(dev_df, preserve_index=False)

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Class weights

labels = train_df["label"].values
class_counts = np.bincount(labels)
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

# Focal loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

# Custom trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Model

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
).to(device)

loss_fn = FocalLoss(weight=class_weights)

# Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds)
    return {"f1": f1}

# Training config

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    loss_fn=loss_fn
)

trainer.train()

# Threshold optimisation

print("\nOptimising threshold...")

predictions = trainer.predict(dev_dataset)
logits = predictions.predictions
labels = predictions.label_ids

probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best Dev F1:", best_f1)

# Save best model
trainer.save_model("BestModel")