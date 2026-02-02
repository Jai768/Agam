import pandas as pd
import numpy as np
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Download NLTK tokenizer for sentence splitting
nltk.download('punkt')

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "mental/mental-bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. DATA PREPARATION
# ==========================================

def load_data(filepath=None):
    """
    Loads data from CSV.
    If no filepath is provided, creates a dummy dataset for demonstration.
    """
    if filepath:
        df = pd.read_csv(filepath)
        # Ensure columns exist
        if 'text' not in df.columns or 'dominant_distortion' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'dominant_distortion' columns")
        return df

    # MOCK DATA (For testing the script)
    print("WARNING: No filepath provided. Generating mock data...")
    data = {
        'text': [
            "I'm a total failure.", "Everyone hates me.", "I should have done better.",
            "If I fail this, my life is over.", "He didn't say hi, he must be angry.",
            "I feel sad, so the situation is hopeless.", "Nothing ever goes right for me.",
            "The weather is nice today.", "I bought some groceries.", "The meeting is at 5 PM."
        ],
        'dominant_distortion': [
            "Labeling", "Mind Reading", "Should Statements",
            "Catastrophizing", "Jumping to Conclusions",
            "Emotional Reasoning", "Overgeneralization",
            "Neutral", "Neutral", "Neutral"  # Crucial: Add 'Neutral' examples!
        ]
    }
    return pd.DataFrame(data)


class DistortionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 3. TRAINING PIPELINE
# ==========================================

def train_model(df):
    # 1. Encode Labels
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['dominant_distortion'])

    # Save label mapping for inference later
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    num_labels = len(le.classes_)

    print(f"Classes found: {label2id}")

    # 2. Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label_id'].values, test_size=0.1, random_state=42
    )

    # 3. Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = DistortionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = DistortionDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 4. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    print("Training Complete.")
    return model, tokenizer, id2label


# ==========================================
# 4. INFERENCE ENGINE (The Request)
# ==========================================

def analyze_journal(text_block, model, tokenizer, id2label):
    """
    Splits text into sentences, predicts distortion, and returns indices.
    """
    # 1. Split into sentences
    sentences = nltk.sent_tokenize(text_block)
    results = []

    model.eval()

    print(f"\nAnalyzing {len(sentences)} sentences...")

    for idx, sentence in enumerate(sentences):
        # Preprocess
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Get best class
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            pred_label = id2label[pred_idx]

        # 2. Logic to filter results
        # We generally only want to flag things that are NOT Neutral and have high confidence
        if pred_label != "Neutral" and confidence > 0.5:
            results.append({
                "sentence_index": idx,
                "text": sentence,
                "distortion": pred_label,
                "confidence": round(confidence, 2)
            })

    return results


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # --- Step A: Load Data ---
    # To use your own data, change this to: load_data("path/to/your_dataset.csv")
    df = load_data()

    # --- Step B: Train ---
    model, tokenizer, id2label = train_model(df)

    # --- Step C: Test Inference ---
    test_journal = """
    I tried to fix the bug today but I couldn't. 
    I am completely incompetent as a developer. 
    Everyone else on the team is smarter than me.
    I will probably get fired tomorrow.
    I went to lunch at 12pm.
    """

    print(f"\n--- Input Text ---\n{test_journal}")

    findings = analyze_journal(test_journal, model, tokenizer, id2label)

    print("\n--- Detected Distortions ---")
    import json

    print(json.dumps(findings, indent=4))

    # Optional: Save the model
    # model.save_pretrained("./cognitive_distortion_model")
    # tokenizer.save_pretrained("./cognitive_distortion_model")