import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# Load data

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
    "label"
]

data["binary_label"] = data["label"].apply(lambda x: 1 if x >= 2 else 0)

print("Dataset shape:", data.shape)
print(data.head())

# Technique 1 - basic statistic profiling

# Class distribution
class_counts = data["binary_label"].value_counts().sort_index()

print("\nClass Counts:")
print(class_counts)

plt.figure(figsize=(6,4))
sns.barplot(
    x=["No PCL", "PCL"],
    y=class_counts.values
)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.show()

print("\nClass Percentages:")
print((class_counts / len(data)) * 100)

# Token length analysis

data["token_count"] = data["paragraph"].astype(str).apply(lambda x: len(x.split()))

print("\nToken Statistics:")
print("Mean:", round(data["token_count"].mean(), 2))
print("Median:", data["token_count"].median())
print("Max:", data["token_count"].max())
print("Min:", data["token_count"].min())

plt.figure(figsize=(8,5))
sns.histplot(data["token_count"], bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.show()

# Token length analysis

data["token_count"] = data["paragraph"].astype(str).apply(lambda x: len(x.split()))

print("\nToken Statistics:")
print("Mean:", round(data["token_count"].mean(), 2))
print("Median:", data["token_count"].median())
print("Max:", data["token_count"].max())
print("Min:", data["token_count"].min())

plt.figure(figsize=(8,5))
sns.histplot(data["token_count"], bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.show()

# Technique 2 - Bigram analysis

positive_texts = data[data["binary_label"] == 1]["paragraph"].astype(str)

vectorizer = CountVectorizer(
    ngram_range=(2,2),
    stop_words="english",
    max_features=2000
)

X = vectorizer.fit_transform(positive_texts)

bigrams = vectorizer.get_feature_names_out()
counts = np.asarray(X.sum(axis=0)).flatten()

bigram_freq = sorted(
    zip(bigrams, counts),
    key=lambda x: x[1],
    reverse=True
)[:20]

bigram_df = pd.DataFrame(bigram_freq, columns=["Bigram", "Frequency"])

print("\nTop 20 Bigrams in PCL Class:")
print(bigram_df)

plt.figure(figsize=(8,6))
sns.barplot(
    data=bigram_df.sort_values("Frequency", ascending=False),
    y="Bigram",
    x="Frequency"
)
plt.title("Top 20 Bigrams in PCL Class")
plt.show()