import openai
import os


class SeverityPredictor:
    def __init__(self):
        openai.api_key = os.getenv("OPEN_AI_KEY")

    def predict(self, symptoms):
        prompt = f"Based on the following symptoms: {symptoms}, suggest a triage level: Critical, Serious, Moderate Mild. reply in 1-2 words.Add plan of action in 1 line."
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
