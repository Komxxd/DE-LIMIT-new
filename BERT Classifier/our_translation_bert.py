import os
import glob
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# 1. Setup Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Configuration (Experiment 3: Translation + BERT Baseline)
LANGUAGE = "French" # The original language (but we will read its translated file)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
# We use standard English BERT here, NOT mBERT, because everything is supposed to be in English now!
MODEL_NAME = "bert-base-uncased" 

# 3. Load Data
def load_translated_data(split):
    # Search for translated datasets
    files = glob.glob(f"../Dataset/{split}/*translated*{LANGUAGE}*.csv")
    if not files:
        raise ValueError(f"No translated files found for {LANGUAGE} in the '{split}' split! Did you run the translation script?")
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list)
    
    label_map = {}
    for label in df['label'].unique():
        if str(label).lower() in ['normal', 'none', '0']:
            label_map[label] = 0
        else:
            label_map[label] = 1 
            
    df['label_id'] = df['label'].map(label_map)
    
    # We use the 'translated' column! If it's missing, fallback to 'text'
    text_col = 'translated' if 'translated' in df.columns else 'text'
    return df[text_col].astype(str).tolist(), df['label_id'].tolist()

print(f"\nLoading translated {LANGUAGE} dataset...")
train_texts, train_labels = load_translated_data("train")
val_texts, val_labels = load_translated_data("val")

# 4. Tokenization
print("Downloading / Loading Standard English BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

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
val_inputs, val_masks, val_labels = encode_data(val_texts, val_labels)

train_dataloader = DataLoader(TensorDataset(train_inputs, train_masks, train_labels), sampler=RandomSampler(train_inputs), batch_size=BATCH_SIZE)
val_dataloader = DataLoader(TensorDataset(val_inputs, val_masks, val_labels), sampler=SequentialSampler(val_inputs), batch_size=BATCH_SIZE)

# 5. Model Setup
print("Downloading / Loading Standard English BERT Brain...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, hidden_dropout_prob=0.3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps/10), num_training_steps=total_steps)

# 6. Training Loop
print("\nStarting Training (Experiment 3: Translation + BERT Baseline)...")
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
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
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
output_dir = f'./saved_model_translated_{LANGUAGE.lower()}/'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
