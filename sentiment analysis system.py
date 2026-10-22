
import tkinter as tk
from tkinter import messagebox
from functools import reduce
import re

import numpy as np
import matplotlib.pyplot as plt


# ---------------- SENTIMENT DICTIONARIES ----------------

positive_words = {
    "good": 1, "great": 2, "excellent": 3,
    "amazing": 3, "love": 3, "loved": 3,
    "lovely": 2, "happy": 2, "useful": 2,
    "best": 3, "wonderful": 3, "nice": 1,
    "perfect": 3, "helpful": 2, "enjoy": 2,
    "enjoyed": 2, "beautiful": 2, "awesome": 3,
    "fantastic": 3, "brilliant": 3, "satisfied": 2,
    "success": 2, "successful": 3, "easy": 1,
    "comfortable": 2, "friendly": 2, "recommend": 2,
    "like": 1, "liked": 2, "impressive": 2,
    "fast": 1, "fresh": 1, "delicious": 2,
    "exciting": 2, "excited": 2, "thankful": 2,
    "valuable": 2, "reliable": 2, "affordable": 2
}

negative_words = {
    "bad": 1, "poor": 2, "terrible": 3,
    "awful": 3, "hate": 3, "hated": 3,
    "sad": 2, "useless": 2, "worst": 3,
    "disappointing": 3, "disappointed": 3,
    "boring": 2, "problem": 1, "difficult": 1,
    "fail": 2, "failed": 2, "failure": 3,
    "angry": 2, "annoying": 2, "slow": 1,
    "expensive": 2, "dirty": 2, "broken": 2,
    "ugly": 2, "painful": 2, "unhappy": 2,
    "useless": 2, "horrible": 3, "dislike": 2,
    "disliked": 2, "waste": 2, "weak": 1,
    "wrong": 1, "unfriendly": 2, "rude": 2,
    "unsafe": 3, "uncomfortable": 2, "regret": 2,
    "poorly": 2, "worried": 1, "frustrating": 2,
    "frustrated": 2, "unreliable": 2
}

NEGATORS = {"not", "never", "no", "hardly", "isn't",
            "wasn't", "don't", "didn't", "can't", "cannot"}


# ---------------- BASE CLASS ----------------

class BaseAnalyzer:

    def preprocess(self, text):
        return re.findall(r"[a-z']+", text.lower())

    def classify(self, score):
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        return "Neutral"


# ---------------- SENTIMENT ANALYZER ----------------

class SimpleSentimentAnalyzer(BaseAnalyzer):

    def __init__(self):
        self.positive_words = positive_words
        self.negative_words = negative_words

    def analyze(self, text):
        words = self.preprocess(text)

        positive_score = 0
        negative_score = 0
        negation_window = 0

        for word in words:

            if word in NEGATORS:
                negation_window = 3
                continue

            weight = 0
            polarity = 0

            if word in self.positive_words:
                weight = self.positive_words[word]
                polarity = 1

            elif word in self.negative_words:
                weight = self.negative_words[word]
                polarity = -1

            if weight:
                if negation_window > 0:
                    polarity *= -1

                if polarity > 0:
                    positive_score += weight
                else:
                    negative_score += weight

            if negation_window > 0:
                negation_window -= 1

        final_score = positive_score - negative_score
        sentiment = self.classify(final_score)

        return {
            "text": text,
            "sentiment": sentiment,
            "score": final_score,
            "positive_score": positive_score,
            "negative_score": negative_score,
            "words_checked": len(words)
        }


# ---------------- NUMPY STATISTICS ----------------

def calculate_statistics(scores):
    values = np.array(scores, dtype=float)

    if len(values) == 0:
        return 0, 0, 0

    return (
        float(np.mean(values)),
        float(np.max(values)),
        float(np.min(values))
    )


# ---------------- REDUCE DEMONSTRATION ----------------

def total_score(values):
    return reduce(lambda a, b: a + b, values, 0)


# ---------------- APPLICATION DATA ----------------

analyzer = SimpleSentimentAnalyzer()
all_results = []


# ---------------- ANALYZE ALL REVIEWS ----------------

def analyze_all():
    global all_results

    raw_text = input_box.get("1.0", tk.END).strip()

    if not raw_text:
        messagebox.showwarning(
            "Input Required",
            "Please enter at least one review."
        )
        return

    # Each non-empty line is treated as one review.
    reviews = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    all_results = [analyzer.analyze(review) for review in reviews]

    counts = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }

    for result in all_results:
        counts[result["sentiment"]] += 1

    scores = [result["score"] for result in all_results]
    mean_score, max_score, min_score = calculate_statistics(scores)

    # Display every review and its result together.
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)

    for i, result in enumerate(all_results, start=1):
        output_box.insert(
            tk.END,
            f"REVIEW {i}\n"
            f"Text: {result['text']}\n"
            f"Sentiment: {result['sentiment']}\n"
            f"Score: {result['score']}\n"
            f"Positive score: {result['positive_score']}\n"
            f"Negative score: {result['negative_score']}\n"
            f"Words checked: {result['words_checked']}\n"
            f"{'-' * 55}\n"
        )

    output_box.insert(
        tk.END,
        "\nOVERALL SENTIMENT SUMMARY\n"
        f"Total reviews: {len(all_results)}\n"
        f"Positive reviews: {counts['Positive']}\n"
        f"Negative reviews: {counts['Negative']}\n"
        f"Neutral reviews: {counts['Neutral']}\n"
        f"Combined score: {total_score(scores)}\n"
        f"Average score: {mean_score:.2f}\n"
        f"Highest score: {max_score:.2f}\n"
        f"Lowest score: {min_score:.2f}\n"
    )

    output_box.config(state="disabled")

    summary_label.config(
        text=(
            f"Reviews: {len(all_results)}   |   "
            f"Positive: {counts['Positive']}   |   "
            f"Negative: {counts['Negative']}   |   "
            f"Neutral: {counts['Neutral']}"
        )
    )


# ---------------- SHOW CHART ----------------

def show_chart():
    if not all_results:
        messagebox.showinfo(
            "No Results",
            "Analyze the reviews before displaying the chart."
        )
        return

    labels = ["Positive", "Negative", "Neutral"]

    counts = [
        sum(r["sentiment"] == label for r in all_results)
        for label in labels
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts)

    plt.title("Sentiment Analysis Results")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.bar_label(bars, padding=3)
    plt.tight_layout()
    plt.show()


# ---------------- CLEAR ALL ----------------

def clear_all():
    global all_results

    input_box.delete("1.0", tk.END)

    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.config(state="disabled")

    all_results = []
    summary_label.config(
        text="Enter reviews below, one review per line."
    )


# ---------------- TKINTER GUI ----------------

root = tk.Tk()
root.title("Sentiment Analysis System")
root.geometry("950x760")
root.minsize(750, 600)

title_label = tk.Label(
    root,
    text="SENTIMENT ANALYSIS SYSTEM",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=10)

instruction_label = tk.Label(
    root,
    text="Enter multiple reviews below (one review per line):",
    font=("Arial", 11)
)
instruction_label.pack()

input_box = tk.Text(
    root,
    height=8,
    width=105,
    font=("Arial", 10),
    wrap="word"
)
input_box.pack(padx=15, pady=8, fill="x")

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

tk.Button(
    button_frame,
    text="Analyze All Reviews",
    command=analyze_all,
    width=20
).grid(row=0, column=0, padx=5)

tk.Button(
    button_frame,
    text="Show Chart",
    command=show_chart,
    width=14
).grid(row=0, column=1, padx=5)

tk.Button(
    button_frame,
    text="Clear All",
    command=clear_all,
    width=12
).grid(row=0, column=2, padx=5)

tk.Button(
    button_frame,
    text="Exit",
    command=root.destroy,
    width=10
).grid(row=0, column=3, padx=5)

summary_label = tk.Label(
    root,
    text="Enter reviews below, one review per line.",
    font=("Arial", 10, "bold"),
    wraplength=900
)
summary_label.pack(pady=8)

output_heading = tk.Label(
    root,
    text="ALL SENTIMENT RESULTS",
    font=("Arial", 12, "bold")
)
output_heading.pack(pady=5)

output_frame = tk.Frame(root)
output_frame.pack(
    padx=15,
    pady=5,
    fill="both",
    expand=True
)

scrollbar = tk.Scrollbar(output_frame)
scrollbar.pack(side="right", fill="y")

output_box = tk.Text(
    output_frame,
    font=("Consolas", 10),
    wrap="word",
    yscrollcommand=scrollbar.set,
    state="disabled"
)
output_box.pack(side="left", fill="both", expand=True)

scrollbar.config(command=output_box.yview)

root.mainloop()
