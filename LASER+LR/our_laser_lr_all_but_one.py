import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers first: pip install sentence-transformers")
    exit()

print("Loading modern Multi-Lingual Dense Embedder (Alternative to legacy LASER)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Configuration (Experiment 6: LASER + LR All But One)
TARGET_LANGUAGE = "German"
TARGET_SAMPLE_SIZE = 16 # Few-shot sample of the target language. Set to 0 for pure zero-shot.

def load_all_but_one_train():
    all_files = glob.glob(f"../Dataset/train/*_full.csv")
    df_list = []
    for f in all_files:
        df = pd.read_csv(f)
        if TARGET_LANGUAGE.lower() in f.lower():
            if TARGET_SAMPLE_SIZE > 0:
                df = df.sample(n=min(TARGET_SAMPLE_SIZE, len(df)), random_state=42)
                df_list.append(df)
        else:
            df_list.append(df)
            
    df = pd.concat(df_list)
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

def load_target_val():
    files = glob.glob(f"../Dataset/val/*{TARGET_LANGUAGE}*_full.csv")
    if not files: return [], []
    df = pd.concat([pd.read_csv(f) for f in files])
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading combined dataset (everything except {TARGET_LANGUAGE})...")
train_texts, train_y = load_all_but_one_train()

print(f"Loading {TARGET_LANGUAGE} validation dataset...")
val_texts, val_y = load_target_val()

print(f"Generating Embeddings for {len(train_texts)} train samples & {len(val_texts)} val samples...")
train_X = model.encode(train_texts, show_progress_bar=True)
val_X = model.encode(val_texts, show_progress_bar=True)

print("\nTraining Logistic Regression Classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(train_X, train_y)

print(f"Evaluating on {TARGET_LANGUAGE}...")
preds = clf.predict(val_X)
f1 = f1_score(val_y, preds, average='macro')
acc = accuracy_score(val_y, preds)

print(f"Validation F1: {f1:.4f} | Accuracy: {acc:.4f}")

import joblib
joblib.dump(clf, f"laser_lr_all_but_one_{TARGET_LANGUAGE.lower()}.pkl")
print(f"Model saved to laser_lr_all_but_one_{TARGET_LANGUAGE.lower()}.pkl")
print("Experiment 6 Complete!")
