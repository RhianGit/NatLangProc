import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    f1_score,
    auc
)

dev_preds = pd.read_csv("dev.txt", header=None)[0].values

dev_labels_df = pd.read_csv("dev_semeval_parids-labels.csv")
dev_labels_df["label"] = dev_labels_df["label"].apply(lambda x: 1 if sum(ast.literal_eval(x)) > 0 else 0)
true_labels = dev_labels_df["label"].values

print("F1 Score:", f1_score(true_labels, dev_preds))
print(classification_report(true_labels, dev_preds, digits=4))

cm = confusion_matrix(true_labels, dev_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No PCL (0)", "PCL (1)"],
            yticklabels=["No PCL (0)", "PCL (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Dev Set)")
plt.show()

precision, recall, thresholds = precision_recall_curve(true_labels, dev_preds)
pr_auc = auc(recall, precision)

false_positives = np.where((dev_preds == 1) & (true_labels == 0))[0]
false_negatives = np.where((dev_preds == 0) & (true_labels == 1))[0]

print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))

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

data["id"] = data["id"].astype(str)

dev_ids = dev_labels_df["par_id"].astype(str).values
dev_text = data[data["id"].isin(dev_ids)].reset_index(drop=True)

print("Sample False Positives:\n")
for i in false_positives[:5]:
    print(dev_text.loc[i, "paragraph"])
    print("-----")

print("\nSample False Negatives:\n")
for i in false_negatives[:5]:
    print(dev_text.loc[i, "paragraph"])
    print("-----")