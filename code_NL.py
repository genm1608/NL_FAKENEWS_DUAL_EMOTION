"""
NIÊN LUẬN NGHIÊN CỨU
Đề tài: Nghiên cứu phát hiện tin giả trên mạng xã hội dựa trên đặc trưng cảm xúc kép

Dataset: PHEME (Rumour Veracity Classification)
Bài toán: Phát hiện tin giả (TRUE vs FALSE)

Ý tưởng chính:
- Kết hợp embedding của source tweet với dual-emotion features
- Trích xuất cảm xúc:
    + Source tweet (cảm xúc nguồn)
    + Replies (phân bố cảm xúc phản hồi)
- Tính dual-emotion gap và thống kê phân bố replies
- Huấn luyện và so sánh 4 mô hình ML
    1. Logistic Regression
    2. SVM (RBF)
    3. Random Forest
    4. XGBoost
- GridSearchCV để tối ưu siêu tham số
- Đo thời gian huấn luyện và biểu đồ so sánh

"""

# ================== IMPORT ==================
import os
import json
import re
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import torch
import time
import matplotlib.pyplot as plt

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ================== CẤU HÌNH ==================
BASE_PATH = os.path.join(os.path.dirname(__file__), "PHEME_veracity")
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ================== TIỀN XỬ LÝ ==================
def clean_text(text):
    """Làm sạch văn bản tweet"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_visible(name):
    """Bỏ qua thư mục ẩn"""
    return isinstance(name, str) and not name.startswith((".", "_"))

# ================== NHÃN VERACITY ==================
def extract_veracity(annotation):
    """0=TRUE, 1=FALSE, None=UNVERIFIED"""
    if not isinstance(annotation, dict):
        return None
    if str(annotation.get("true", "")).strip() == "1":
        return 0
    if str(annotation.get("misinformation", "")).strip() == "1":
        return 1
    return None

# ================== LOAD DATASET ==================
def load_pheme(base_path):
    """Đọc toàn bộ dataset PHEME"""
    records = []
    events = [e for e in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, e)) and is_visible(e)]
    print("📂 Các sự kiện:", events)

    for event in tqdm(events, desc="Đọc sự kiện"):
        event_path = os.path.join(base_path, event)
        rumours_path = os.path.join(event_path, "rumours")
        if not os.path.isdir(rumours_path):
            continue
        for thread_id in os.listdir(rumours_path):
            if not is_visible(thread_id):
                continue
            tpath = os.path.join(rumours_path, thread_id)
            ann_path = os.path.join(tpath, "annotation.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            label = extract_veracity(ann)
            if label is None:
                continue

            # -------- source tweet --------
            source_text = ""
            for root, _, files in os.walk(tpath):
                for fn in files:
                    if fn.endswith(".json") and not fn.startswith(("annotation", "structure")):
                        try:
                            with open(os.path.join(root, fn), "r", encoding="utf-8") as f:
                                j = json.load(f)
                            source_text = j.get("text") or j.get("tweet_text") or ""
                            if source_text:
                                break
                        except:
                            pass
                if source_text:
                    break
            if not source_text:
                continue

            # -------- replies --------
            replies = []
            rdir = os.path.join(tpath, "reactions")
            if os.path.isdir(rdir):
                for fn in os.listdir(rdir):
                    if fn.endswith(".json"):
                        try:
                            with open(os.path.join(rdir, fn), "r", encoding="utf-8") as f:
                                j = json.load(f)
                            txt = j.get("text") or ""
                            if txt:
                                replies.append(txt)
                        except:
                            pass

            records.append({"event": event, "thread_id": thread_id, "source_text": source_text, "replies": replies, "label": label})

    df = pd.DataFrame(records)
    print("📊 Phân bố nhãn:", Counter(df["label"]))
    df["source_text_clean"] = df["source_text"].apply(clean_text)
    df["replies_clean"] = df["replies"].apply(lambda L: [clean_text(x) for x in L])
    return df

# ================== EMOTION ==================
def get_emotions(texts, pipe, batch_size=32):
    """Dự đoán cảm xúc theo batch"""
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = pipe(batch, truncation=True, padding=True)
        outputs.extend([o['label'].lower() for o in out])
    return outputs

def reply_distribution(replies, pipe):
    """Phân bố cảm xúc replies + thống kê"""
    dist = {e: 0.0 for e in EMOTION_LABELS}
    if not replies:
        dist['neutral'] = 1.0
        return dist
    emos = get_emotions(replies, pipe)
    cnt = Counter(emos)
    total = len(emos)
    for e, c in cnt.items():
        if e in dist:
            dist[e] = c / total
    return dist

# ================== FEATURE ==================
def build_features(src_emo, reply_dist, embeddings):
    """Kết hợp dual-emotion + embedding + thống kê replies"""
    # Source one-hot
    src_df = pd.get_dummies(pd.Series(src_emo), prefix="src")
    for e in EMOTION_LABELS:
        col = f"src_{e}"
        if col not in src_df.columns:
            src_df[col] = 0
    src_df = src_df[[f"src_{e}" for e in EMOTION_LABELS]]

    # Reply distribution + statistics
    rep_df = pd.DataFrame(reply_dist).fillna(0)
    for e in EMOTION_LABELS:
        if e not in rep_df.columns:
            rep_df[e] = 0
    rep_df = rep_df[EMOTION_LABELS]

    # Gap
    gap = src_df.values - rep_df.values

    # Embedding
    X = np.hstack([embeddings, src_df.values, rep_df.values, gap])

    return X

# ================== MAIN ==================
def main():
    print("=== LOAD DATA ===")
    df = load_pheme(BASE_PATH)

    # Device cho pipeline
    device = 0 if torch.cuda.is_available() else -1
    emo_pipe = pipeline("text-classification", model=EMOTION_MODEL, device=device)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')

    print("=== TRÍCH XUẤT CẢM XÚC ===")
    # Source emotions
    src_emo = get_emotions(df["source_text_clean"].tolist(), emo_pipe)
    # Reply distribution
    rep_dist = [reply_distribution(r, emo_pipe) for r in tqdm(df["replies_clean"], desc="Replies")]

    print("=== TÍNH EMBEDDING SOURCE ===")
    embeddings = embed_model.encode(df["source_text_clean"].tolist(), batch_size=32, show_progress_bar=True)

    print("=== BUILD FEATURES ===")
    X = build_features(src_emo, rep_dist, embeddings)
    y = df["label"].values

    # Split và SMOTE
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_tr, y_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)

    # ================== MÔ HÌNH ==================
    models = {
        "LogisticRegression": {"model": LogisticRegression(max_iter=1000), "params": {"clf__C": [0.01, 0.1, 1, 10]}},
        "SVM_RBF": {"model": SVC(kernel="rbf"), "params": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]}},
        "RandomForest": {"model": RandomForestClassifier(), "params": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10, 20]}},
        "XGBoost": {"model": XGBClassifier(eval_metric="logloss", use_label_encoder=False), "params": {"clf__n_estimators": [100, 300],
                    "clf__max_depth": [3,6], "clf__learning_rate": [0.01, 0.1]}}
    }

    results = {}

    for name, cfg in models.items():
        print(f"\n===== {name} =====")
        pipe_clf = Pipeline([("scaler", StandardScaler()), ("clf", cfg["model"])])
        grid = GridSearchCV(pipe_clf, cfg["params"], cv=5, scoring="f1", n_jobs=-1)

        start_time = time.time()
        grid.fit(X_tr, y_tr)
        elapsed = time.time() - start_time

        y_pred = grid.best_estimator_.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)
        rec = recall_score(y_te, y_pred)

        results[name] = {"accuracy": acc, "f1": f1, "recall": rec, "time": elapsed}

        print(f"Time to train {name}: {elapsed:.2f} seconds")
        print("Accuracy:", acc, "F1-score:", f1, "Recall:", rec)
        print("Best params:", grid.best_params_)
        print(classification_report(y_te, y_pred, target_names=["True", "False"]))

    # ================== VẼ BIỂU ĐỒ ==================
    models_list = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models_list]
    f1_score_list = [results[m]["f1"] for m in models_list]
    recall_list = [results[m]["recall"] for m in models_list]
    time_list = [results[m]["time"] for m in models_list]

    x = np.arange(len(models_list))
    width = 0.2

    # Biểu đồ Accuracy, F1, Recall
    plt.figure(figsize=(12,6))
    plt.bar(x - width, accuracy, width, label="Accuracy", color='skyblue')
    plt.bar(x, f1_score_list, width, label="F1-score", color='salmon')
    plt.bar(x + width, recall_list, width, label="Recall", color='lightgreen')
    plt.xticks(x, models_list)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("So sánh Accuracy, F1-score, Recall giữa các mô hình")
    plt.legend()
    plt.show()

    # Biểu đồ thời gian huấn luyện
    plt.figure(figsize=(10,5))
    plt.bar(models_list, time_list, color='orchid')
    plt.ylabel("Time (seconds)")
    plt.title("So sánh thời gian huấn luyện các mô hình")
    plt.show()

if __name__ == "__main__":
    main()