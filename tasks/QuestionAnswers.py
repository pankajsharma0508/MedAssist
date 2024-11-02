from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="./dynamic_tinybert_symptoms_to_disease",
    tokenizer="Intel/dynamic_tinybert",
)

# Example usage
context = "I may have itching,vomiting,fatigue,weight_loss,high_fever,yellowish_skin,dark_urine,abdominal_pain"
question = "what condition do I have?"
result = qa_pipeline(question=question, context=context)
print(result)
