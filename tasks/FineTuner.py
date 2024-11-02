from datasets import load_dataset, Dataset
import pandas as pd
import numpy
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

model_name = "Intel/dynamic_tinybert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
fineTuneModelName = "dynamic_tinybert_symptoms_to_disease"

df = pd.read_json(
    "hf://datasets/fhai50032/Symptoms_to_disease_7k/Symptoms_to_disease_7k.json"
)

query = df["query"].apply(lambda x: x.split("Patient:")[1].strip())
response = df["response"]

df["context"] = f"{query}, In this case it is possible that {response}."
df["question"] = "What condition do these symptoms suggest?"

selected_entries = df.head(10)
huggingface_dataset = Dataset.from_pandas(selected_entries)


def preprocess(example):
    # Tokenize the input and align it for question answering
    inputs = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )
    inputs["start_positions"] = 0
    inputs["end_positions"] = len(example["response"]) - 1
    return inputs


tokenized_dataset = huggingface_dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

print(tokenized_dataset)

# Initialize Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

# Train the model
trainer.train()

trainer.save_model("./" + fineTuneModelName)

# model = TinyBertForSequenceClassification.from_pretrained("./" + fineTuneModelName)
# tokenizer = TinyBertTokenizer.from_pretrained("./" + fineTuneModelName)

# model_name = "PankSharma/" + fineTuneModelName

# # Push the model to Hugging Face Hub
# model.push_to_hub(model_name)
# tokenizer.push_to_hub(model_name)
