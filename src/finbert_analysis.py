import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load FinBERT once globally
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_finbert_sentiment(suffix: str) -> float:
    """
    Perform FinBERT sentiment analysis on a preprocessed transcript file.
    Reads from ready_for_nlp_<suffix>.txt and saves to finbert_sentiment_output_<suffix>.csv.

    Args:
        suffix (str): Suffix for input/output files (e.g., 'current', 'prev1')

    Returns:
        float: Average FinBERT sentiment score
    """
    # Construct input and output file names
    input_file = f"ready_for_nlp_{suffix}.txt"
    output_file = f"finbert_sentiment_output_{suffix}.csv"

    # Load transcript
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract only sentences
    sentences = [line.split(":", 1)[1].strip() for line in lines if ":" in line]

    # Create Dataset
    data = Dataset.from_dict({"text": sentences})

    # Tokenize
    def tokenize_fn(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    data = data.map(tokenize_fn, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Predict in batches
    batch_size = 32
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            scores, labels = torch.max(probs, dim=1)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to readable labels
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    df = pd.DataFrame({
        "index": np.arange(1, len(sentences)+1),
        "sentence": sentences,
        "label": [label_map[l] for l in all_labels],
        "score": np.asarray(all_scores)  # Use np.asarray to avoid copy issue
    })

    # Save FinBERT CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Saved FinBERT output to {output_file}")

    # Calculate average sentiment score
    score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    avg_sentiment = df["label"].map(score_map).mean()
    print(f"📈 Average FinBERT Sentiment: {avg_sentiment:.4f}")

    return float(avg_sentiment)
avg_finbert_current = run_finbert_sentiment("current")
avg_finbert_prev1 = run_finbert_sentiment("prev1")
avg_finbert_prev2 = run_finbert_sentiment("prev2")
avg_finbert_prev3 = run_finbert_sentiment("prev3")
def run_all_finbert():
    return {
        "current": run_finbert_sentiment("current"),
        "prev1": run_finbert_sentiment("prev1"),
        "prev2": run_finbert_sentiment("prev2"),
        "prev3": run_finbert_sentiment("prev3"),
    }

