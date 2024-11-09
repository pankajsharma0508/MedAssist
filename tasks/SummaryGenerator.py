import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import time
from rouge_score import rouge_scorer

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"


class SummaryGenerator:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=50,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        # Decode and print the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Summary:", summary)
        return summary

    def evaluate_metrics(self):
        """
        Evaluates the summarization model using ROUGE, latency, and throughput metrics.
        """
        X_test = ["This is a test sentence.", "Here is another sentence to test."]
        y_test = [
            "Test sentence summary.",
            "Test summary of another sentence.",
        ]
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Start measuring inference time (latency)
        start_time = time.time()

        # Generate predictions
        summaries = []
        for text in X_test:
            summary = self.summarize(text)
            summaries.append(summary)

        # End measuring inference time
        end_time = time.time()

        # Latency (in ms)
        latency = (
            (end_time - start_time) * 1000 / len(X_test)
        )  # Average latency per request in milliseconds
        throughput = len(X_test) / (end_time - start_time)  # Requests per second

        # Calculate ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref, pred in zip(y_test, summaries):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        # Average ROUGE scores
        rouge1 = np.mean(rouge_scores["rouge1"])
        rouge2 = np.mean(rouge_scores["rouge2"])
        rougeL = np.mean(rouge_scores["rougeL"])

        return self.model, latency, throughput, rouge1, rouge2, rougeL


print(SummaryGenerator().evaluate_metrics())
