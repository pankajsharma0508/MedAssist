import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the DistilBERT model and tokenizer from Hugging Face
model_name = "distilbert-base-uncased"  # Change to "microsoft/Phi-3.5-mini-instruct" if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)  # Set num_labels to the number of document types

# Define document types (modify as needed)
labels = {0: "Radiology Report", 1: "Discharge Summary", 2: "Lab Report"}

# Example clinical documents
documents = [
    "Patient presented with chest pain, underwent CT scan showing abnormalities in the left lung...",
    "Patient discharged with medications for hypertension, follow-up scheduled in two weeks...",
    "Blood test results show elevated white blood cell count, suggesting possible infection...",
]

# Tokenize and classify each document
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Run classification
for doc in documents:
    prediction = classifier(doc)[0]
    label_id = torch.argmax(torch.tensor(prediction["score"]))
    document_type = labels[label_id.item()]
    print(f"Document: {doc[:50]}...")  # Display a snippet of the document
    print(f"Predicted Type: {document_type}\n")
