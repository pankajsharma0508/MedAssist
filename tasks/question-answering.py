# This service is responsible for gathering information from user
# Helping medical professionals to dignose the problem based on user inputs.

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForQuestionAnswering.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1"
)

# Sample context and question
context = "The patient shows symptoms of respiratory distress and mild fever."
question = "What are the symptoms of the patient?"

# Tokenize input
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Extract the answer
answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax() + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)

print("Answer:", answer)
