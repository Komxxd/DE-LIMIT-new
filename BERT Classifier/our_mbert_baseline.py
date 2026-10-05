import os
import glob
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# 1. Setup Device (Optimized for Mac and standard GPUs)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps") # Apple Silicon!
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Configuration (Using Standard Optimal Hyperparameters)
LANGUAGE = "German" # Feel free to change this to 'French', 'Arabic', etc.
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
MODEL_NAME = "bert-base-multilingual-cased" # Direct from HuggingFace Hub

# 3. Load Data
def load_data(split):
    # Search the Dataset folder for the requested language
    files = glob.glob(f"../Dataset/{split}/*{LANGUAGE}*_full.csv")
    if not files:
        raise ValueError(f"No files found for {LANGUAGE} in the '{split}' split!")
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list)
    
    # Map string labels to binary integers dynamically
    label_map = {}
    for label in df['label'].unique():
        if str(label).lower() in ['normal', 'none', '0']:
            label_map[label] = 0
        else:
            label_map[label] = 1 # Hate Speech / Abusive
            
    df['label_id'] = df['label'].map(label_map)
    return df['text'].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading {LANGUAGE} dataset...")
train_texts, train_labels = load_data("train")
val_texts, val_labels = load_data("val")

# 4. Tokenization
print("Downloading / Loading mBERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

def encode_data(texts, labels):
    encoded_dict = tokenizer(
        texts,                      
        add_special_tokens=True, 
        max_length=MAX_LEN,           
        padding='max_length',
        truncation=True,
        return_attention_mask=True,   
        return_tensors='pt',     
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask'], torch.tensor(labels)

train_inputs, train_masks, train_labels = encode_data(train_texts, train_labels)

# Calculate Class Weights to combat dataset imbalance
num_neg = (train_labels == 0).sum().item()
num_pos = (train_labels == 1).sum().item()
total = len(train_labels)
class_weights = torch.tensor([total / (2.0 * num_neg), total / (2.0 * num_pos)], dtype=torch.float).to(device)

val_inputs, val_masks, val_labels = encode_data(val_texts, val_labels)

train_dataloader = DataLoader(
    TensorDataset(train_inputs, train_masks, train_labels),
    sampler=RandomSampler(train_inputs),
    batch_size=BATCH_SIZE
)

val_dataloader = DataLoader(
    TensorDataset(val_inputs, val_masks, val_labels),
    sampler=SequentialSampler(val_inputs),
    batch_size=BATCH_SIZE
)

# 5. Model Setup
print("Downloading / Loading mBERT Brain...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, hidden_dropout_prob=0.3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps/10), num_training_steps=total_steps)

# 6. Training Loop
print("\nStarting Training (Experiment 1: mBERT Baseline)...")
for epoch_i in range(0, EPOCHS):
    print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print(f"  Batch {step:>5,}  of  {len(train_dataloader):>5,}.")
            
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        # Calculate custom loss using our class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(outputs.logits.view(-1, 2), b_labels.view(-1))
        
        total_train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print(f"Average training loss: {total_train_loss / len(train_dataloader):.2f}")

    # Validation Phase
    model.eval()
    preds, true_labels = [], []
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        preds.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())
        
    f1 = f1_score(true_labels, preds, average='macro')
    acc = accuracy_score(true_labels, preds)
    print(f"Validation F1-Score: {f1:.4f} | Accuracy: {acc:.4f}")

print("\nTraining complete! Saving model...")
output_dir = f'./saved_model_{LANGUAGE.lower()}/'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
