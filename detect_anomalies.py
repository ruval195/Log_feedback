# -------------------------
# Copyriht
# -------------------------

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


LOG_FILE = "loghub-2.0/2k_dataset/Linux/Linux_2k.log"

with open(LOG_FILE, "r") as f:
    raw_logs = [line.strip() for line in f if line.strip()]

np.random.seed(42)
true_labels = np.ones(len(raw_logs), dtype=int)
anomalous_indices = np.random.choice(len(raw_logs), size=int(0.1 * len(raw_logs)), replace=False)
true_labels[anomalous_indices] = -1

# -------------------------
# Word2Vec + Isolation Forest
# -------------------------
tokenized_logs = [log.lower().split() for log in raw_logs]
w2v_model = Word2Vec(sentences=tokenized_logs, vector_size=50, window=3, min_count=1, workers=1, epochs=20)
w2v_vectors = [
    np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv], axis=0)
    if words else np.zeros(50)
    for words in tokenized_logs
]

iso_w2v = IsolationForest(random_state=42, contamination=0.1)
pred_w2v = iso_w2v.fit_predict(w2v_vectors)

# -------------------------
# LogBERT-style embedding
# -------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_bert_embedding(log_line):
    inputs = tokenizer(log_line, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

bert_vectors = [get_bert_embedding(log) for log in tqdm(raw_logs, desc="LogBERT Embedding")]

iso_bert = IsolationForest(random_state=42, contamination=0.1)
pred_bert = iso_bert.fit_predict(bert_vectors)

# -------------------------
# Metrics
# -------------------------
df = pd.DataFrame({
    "log": raw_logs,
    "true_label": true_labels,
    "pred_word2vec": pred_w2v,
    "pred_logbert": pred_bert
})

# -------------------------
# Output
# -------------------------

# Logs Word2Vec
anomalies_w2v = df[df["pred_word2vec"] == -1][["log", "true_label"]]
anomalies_w2v.to_csv("anomalies_word2vec_linux2k.csv", index=False)

# Logs LogBERT
anomalies_bert = df[df["pred_logbert"] == -1][["log", "true_label"]]
anomalies_bert.to_csv("anomalies_logbert_linux2k.csv", index=False)

report_w2v = classification_report(true_labels, pred_w2v)
report_bert = classification_report(true_labels, pred_bert)

with open("metrics_word2vec_linux2k.txt", "w") as f:
    f.write("Word2Vec + Isolation Forest Metrics\n")
    f.write(report_w2v)

with open("metrics_logbert_linux2k.txt", "w") as f:
    f.write("LogBERT\n")
    f.write(report_bert)
