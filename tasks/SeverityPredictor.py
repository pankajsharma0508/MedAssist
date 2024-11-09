import numpy as np
import openai
import os
import time
from rouge_score import rouge_scorer


class SeverityPredictor:
    def __init__(self):
        openai.api_key = os.getenv("OPEN_AI_KEY")
        self.model = "gpt-3.5-turbo"

    def predict(self, symptoms):
        prompt = f"Based on the following symptoms: {symptoms}, suggest a triage level: Critical, Serious, Moderate Mild. reply in 1-2 words.Add plan of action in 1 line."
        response = openai.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def evaluate_metrics(self):
        """
        Evaluates the model using accuracy, latency, F1 score, and ROUGE scores.
        """
        # Sample test data (symptoms and labeled severity levels)
        X_test = [
            "Shortness of breath, chest pain",
            "Mild headache, slight fatigue",
            "High fever, persistent cough",
            "Minor cut on finger, no bleeding",
        ]
        y_test = ["Critical", "Mild", "Serious", "Mild"]  # Expected severity levels
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        # Start measuring inference time (latency)
        start_time = time.time()

        # Generate predictions
        y_pred = []
        for symptoms in X_test:
            prediction = self.predict(symptoms)
            y_pred.append(prediction.strip())  # Add the predicted triage level

        # End measuring inference time
        end_time = time.time()

        # Calculate latency and throughput
        latency = (
            (end_time - start_time) * 1000 / len(X_test)
        )  # Average latency per request in ms
        throughput = len(X_test) / (end_time - start_time)  # Requests per second

        # Calculate ROUGE scores
        rouge1_scores, rougeL_scores = [], []
        for ref, pred in zip(y_test, y_pred):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        # Average ROUGE scores
        rouge1_avg = np.mean(rouge1_scores)
        rougeL_avg = np.mean(rougeL_scores)

        return self.model, latency, throughput, rouge1_avg, rougeL_avg


# print(SeverityPredictor().evaluate_metrics())
