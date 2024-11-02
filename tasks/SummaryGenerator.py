from transformers import BartForConditionalGeneration, BartTokenizer
import torch

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
