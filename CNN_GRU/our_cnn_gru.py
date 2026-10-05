import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from collections import Counter

# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Config (Experiment 4: CNN+GRU)
LANGUAGE = "German"
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001
MAX_LEN = 128
VOCAB_SIZE = 10000
EMBED_DIM = 300
HIDDEN_DIM = 128

# 3. Load Data
def load_data(split):
    files = glob.glob(f"../Dataset/{split}/*{LANGUAGE}*_full.csv")
    if not files: return [], []
    df = pd.concat([pd.read_csv(f) for f in files])
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['normal', 'none', '0'] else 1)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading {LANGUAGE} dataset...")
train_texts, train_labels = load_data("train")
val_texts, val_labels = load_data("val")

# 3b. Load MUSE Embeddings (Exactly like the original paper)
lang_map = {'Arabic': 'ar', 'French': 'fr', 'Portugese': 'pt', 'Spanish': 'es', 'English': 'en', 'Indonesian': 'id', 'Italian': 'it', 'German': 'de', 'Polish': 'pl'}
lang_code = lang_map[LANGUAGE]
muse_path = f"muse_embeddings/wiki.multi.{lang_code}.vec"

if not os.path.exists(muse_path):
    raise FileNotFoundError(f"MUSE embeddings not found at {muse_path}! Please download them from https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.{lang_code}.vec and place them in the muse_embeddings/ folder.")

print("Loading Official MUSE Embeddings (this takes a moment)...")
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with open(emb_path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            try:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                if word not in word2id:
                    vectors.append(vect)
                    word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
            except ValueError:
                continue
    return np.vstack(vectors), word2id

vector, word2idx = load_vec(muse_path)
pad_vec = np.zeros((1, 300))
unk_vec = np.random.randn(1, 300)
merged_vec = np.vstack((vector, unk_vec, pad_vec))
VOCAB_SIZE = merged_vec.shape[0]

word2idx['<UNK>'] = VOCAB_SIZE - 2
word2idx['<PAD>'] = VOCAB_SIZE - 1
pretrained_embeddings = torch.tensor(merged_vec, dtype=torch.float32)

def encode(texts):
    seqs = []
    for t in texts:
        tokens = [word2idx.get(w, word2idx['<UNK>']) for w in t.lower().split()]
        seqs.append(tokens[:MAX_LEN] + [word2idx['<PAD>']] * max(0, MAX_LEN - len(tokens)))
    return torch.tensor(seqs)

train_inputs = encode(train_texts)
val_inputs = encode(val_texts)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

train_loader = DataLoader(TensorDataset(train_inputs, train_labels), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

# 4. CNN + GRU Architecture
class CNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=word2idx['<PAD>'])
        self.embed.weight.data.copy_(pretrained_embeddings)
        self.conv = nn.Conv1d(EMBED_DIM, 64, kernel_size=3, padding=1)
        self.gru = nn.GRU(64, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)
        
    def forward(self, x):
        x = self.embed(x) 
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.conv(x)) 
        x = x.permute(0, 2, 1) 
        _, hidden = self.gru(x)
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(out)

model = CNN_GRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 5. Training Loop
print(f"\nStarting Training (Experiment 4: CNN+GRU)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x.to(device))
        loss = criterion(out, batch_y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    model.eval()
    preds, true_y = [], []
    for batch_x, batch_y in val_loader:
        with torch.no_grad():
            out = model(batch_x.to(device))
            preds.extend(out.argmax(dim=1).cpu().numpy())
            true_y.extend(batch_y.numpy())
            
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} | Val F1: {f1_score(true_y, preds, average='macro'):.4f} | Val Acc: {accuracy_score(true_y, preds):.4f}")

print("\nDone! CNN_GRU architecture successfully trained.")
torch.save(model.state_dict(), f"cnn_gru_{LANGUAGE.lower()}.pt")
print(f"Model weights saved to cnn_gru_{LANGUAGE.lower()}.pt")
