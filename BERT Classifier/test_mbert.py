import os, glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
LANGUAGE = "German"
MAX_LEN = 128
BATCH_SIZE = 16

# Change this if you want to test the "All But One" model instead!
MODEL_DIR = f"./saved_model_{LANGUAGE.lower()}/" 

def load_test_data():
    files = glob.glob(f"../Dataset/test/*{LANGUAGE}*_full.csv")
    if not files: raise ValueError("No test files found!")
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list)
    label_map = {}
    for label in df['label'].unique():
        if str(label).lower() in ['normal', 'none', '0']: label_map[label] = 0
        else: label_map[label] = 1 
    df['label_id'] = df['label'].map(label_map)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading Test Data for {LANGUAGE}...")
test_texts, test_labels = load_test_data()

print(f"Loading Saved mBERT Model from {MODEL_DIR}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print("Tokenizing Test Data...")
encoded_dict = tokenizer(
    test_texts, add_special_tokens=True, max_length=MAX_LEN, 
    padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
)
test_dataloader = DataLoader(
    TensorDataset(encoded_dict['input_ids'], encoded_dict['attention_mask'], torch.tensor(test_labels)), 
    sampler=SequentialSampler(encoded_dict['input_ids']), batch_size=BATCH_SIZE
)

print("Running Final Evaluation on Held-Out Test Set...\n")
preds, true_labels = [], []
for batch in test_dataloader:
    with torch.no_grad():
        outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=batch[1].to(device))
    preds.extend(np.argmax(outputs.logits.detach().cpu().numpy(), axis=1).flatten())
    true_labels.extend(batch[2].to('cpu').numpy().flatten())

print("================ FINAL RESULTS ================")
print(f"Test F1-Score: {f1_score(true_labels, preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(true_labels, preds):.4f}")
print("===============================================")
print("\nDetailed Classification Report:")
print(classification_report(true_labels, preds, target_names=["Normal", "Hate Speech"]))
