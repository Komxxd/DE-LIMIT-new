import glob, os, torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

LANGUAGE = "German"
MAX_LEN = 128
BATCH_SIZE = 32
MODEL_FILE = f"cnn_gru_{LANGUAGE.lower()}.pt"

def load_data(split):
    files = glob.glob(f"../Dataset/{split}/*{LANGUAGE}*_full.csv")
    if not files: raise ValueError("No test files found!")
    df = pd.concat([pd.read_csv(f) for f in files])
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print("\nRebuilding exact vocabulary map from MUSE...")
def load_vec(emb_path, nmax=50000):
    vectors, word2id = [], {}
    with open(emb_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            try:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if word not in word2id:
                    vectors.append(vect)
                    word2id[word] = len(word2id)
                if len(word2id) == nmax: break
            except ValueError: continue
    return np.vstack(vectors), word2id

lang_map = {'German': 'de', 'French': 'fr', 'Portugese': 'pt', 'Spanish': 'es', 'English': 'en', 'Indonesian': 'id', 'Italian': 'it', 'Polish': 'pl', 'Arabic': 'ar'}
_, word2idx = load_vec(f"muse_embeddings/wiki.multi.{lang_map[LANGUAGE]}.vec")
VOCAB_SIZE = len(word2idx) + 2
word2idx['<UNK>'] = VOCAB_SIZE - 2
word2idx['<PAD>'] = VOCAB_SIZE - 1

print(f"Loading Test Data for {LANGUAGE}...")
test_texts, test_labels = load_data("test")

def encode(texts):
    seqs = []
    for t in texts:
        tokens = [word2idx.get(w, word2idx['<UNK>']) for w in t.lower().split()]
        seqs.append(tokens[:MAX_LEN] + [word2idx['<PAD>']] * max(0, MAX_LEN - len(tokens)))
    return torch.tensor(seqs)

test_inputs = encode(test_texts)
test_loader = DataLoader(TensorDataset(test_inputs, torch.tensor(test_labels)), batch_size=BATCH_SIZE)

class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, 300, padding_idx=word2idx['<PAD>'])
        self.conv = nn.Conv1d(300, 64, kernel_size=3, padding=1)
        self.gru = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x)).permute(0, 2, 1)
        _, hidden = self.gru(x)
        return self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Saved CNN+GRU Model from {MODEL_FILE}...")
model = CNN_GRU().to(device)
# We use weights_only=True for security against malicious pickles
model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
model.eval()

print("Running Final Evaluation on Held-Out Test Set...\n")
preds, true_y = [], []
for batch_x, batch_y in test_loader:
    with torch.no_grad():
        out = model(batch_x.to(device))
        preds.extend(out.argmax(dim=1).cpu().numpy())
        true_y.extend(batch_y.numpy())

print("================ FINAL RESULTS ================")
print(f"Test F1-Score: {f1_score(true_y, preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(true_y, preds):.4f}")
print("===============================================")
print("\nDetailed Classification Report:")
print(classification_report(true_y, preds, target_names=["Normal", "Hate Speech"]))
