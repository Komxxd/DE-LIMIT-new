import glob, joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

LANGUAGE = "German"
# Change this if you want to test the "All But One" model instead!
MODEL_FILE = f"laser_lr_baseline_{LANGUAGE.lower()}.pkl"

def load_test_data():
    files = glob.glob(f"../Dataset/test/*{LANGUAGE}*_full.csv")
    if not files: raise ValueError("No test files found!")
    df = pd.concat([pd.read_csv(f) for f in files])
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading Test Data for {LANGUAGE}...")
test_texts, test_y = load_test_data()

print("Generating Embeddings for Test Set...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
test_X = embedder.encode(test_texts, show_progress_bar=True)

print(f"\nLoading saved Logistic Regression model from {MODEL_FILE}...")
clf = joblib.load(MODEL_FILE)

print("Running Final Evaluation on Held-Out Test Set...\n")
preds = clf.predict(test_X)

print("================ FINAL RESULTS ================")
print(f"Test F1-Score: {f1_score(test_y, preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(test_y, preds):.4f}")
print("===============================================")
print("\nDetailed Classification Report:")
print(classification_report(test_y, preds, target_names=["Normal", "Hate Speech"]))
