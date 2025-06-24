import numpy as np
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

def run_vader_sentiment_analysis(suffix):
    # Construct input and output file names
    input_file = f"ready_for_nlp_{suffix}.txt"
    output_file = f"vader_sentiment_output_{suffix}.csv"

    # Load cleaned transcript
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Break into sentences
    sentences_current = sent_tokenize(text)

    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    # Store scores
    vader_results_current = []

    for i, sentence in enumerate(sentences_current):
        score = analyzer.polarity_scores(sentence)
        vader_results_current.append({
            "index": i+1,
            "sentence": sentence,
            "pos": score['pos'],
            "neu": score['neu'],
            "neg": score['neg'],
            "compound": score['compound']
        })

    # Calculate average compound score
    avg_sentiment_current = np.mean([s['compound'] for s in vader_results_current])
    print(f"\n Average VADER Sentiment_{suffix}(compound): {avg_sentiment_current:.4f}")

    # Optional: Save to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=vader_results_current[0].keys())
        writer.writeheader()
        writer.writerows(vader_results_current)

    print(f" All sentence-level sentiment scores saved to {output_file}")
    return avg_sentiment_current  
try:
    avg_vader_current = run_vader_sentiment_analysis("current")
    avg_vader_prev1 = run_vader_sentiment_analysis("prev1")
    avg_vader_prev2 = run_vader_sentiment_analysis("prev2")
    avg_vader_prev3 = run_vader_sentiment_analysis("prev3")
except Exception as e:
    print(f"Error during sentiment analysis: {str(e)}")
def compute_vader_deltas(current, prev1, prev2, prev3):
    if None not in [current, prev1, prev2, prev3]:
        delta1 = current - prev1
        delta2 = prev1 - prev2
        delta3 = prev2 - prev3
        return delta1, delta2, delta3
    else:
        raise ValueError("❌ One or more sentiment values are missing (None).")

