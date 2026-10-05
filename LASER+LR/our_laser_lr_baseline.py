import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Note: The original researchers used Facebook's LASER embeddings (which require compiling C++ binaries).
# We are upgrading this to use HuggingFace's "sentence-transformers" which is the modern Python-native equivalent!
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers first: pip install sentence-transformers")
    exit()

print("Loading modern Multi-Lingual Dense Embedder (Alternative to legacy LASER)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Configuration (Experiment 5: LASER + LR Baseline)
LANGUAGE = "German"

def load_data(split):
    files = glob.glob(f"../Dataset/{split}/*{LANGUAGE}*_full.csv")
    if not files: return [], []
    df = pd.concat([pd.read_csv(f) for f in files])
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading {LANGUAGE} dataset...")
train_texts, train_y = load_data("train")
val_texts, val_y = load_data("val")

print("Generating Multilingual Sentence Embeddings (this may take a moment)...")
train_X = model.encode(train_texts, show_progress_bar=True)
val_X = model.encode(val_texts, show_progress_bar=True)

print("\nTraining Logistic Regression Classifier on Dense Embeddings...")
clf = LogisticRegression(max_iter=1000)
clf.fit(train_X, train_y)

print("Evaluating...")
preds = clf.predict(val_X)
f1 = f1_score(val_y, preds, average='macro')
acc = accuracy_score(val_y, preds)

print(f"Validation F1: {f1:.4f} | Accuracy: {acc:.4f}")

import joblib
joblib.dump(clf, f"laser_lr_baseline_{LANGUAGE.lower()}.pkl")
print(f"Model saved to laser_lr_baseline_{LANGUAGE.lower()}.pkl")
print("Experiment 5 Complete!")
