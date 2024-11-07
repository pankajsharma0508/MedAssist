import openai
import time
import os

openai.api_key = os.getenv("OPEN_AI_KEY")


class FineTuner:
    def __init__(self):
        self.file_path = "Symptoms_to_disease.jsonl"

    # Step 1: Upload the file
    def upload_file(self):
        response = openai.files.create(
            file=open(self.file_path, "rb"), purpose="fine-tune"
        )
        self.file_id = response.id
        print(f"File uploaded successfully. File ID: {self.file_id}")

    # Step 2: Start fine-tuning
    def fine_tune(self):
        self.fine_tune_response = openai.fine_tunes.create(
            model="gpt-3.5-turbo-0125", training_file=self.file_id
        )
        self.fine_tune_id = self.fine_tune_response.id
        print(f"Fine-tuning started. Fine-tune ID: {self.fine_tune_id}")

        # Step 3: Monitor fine-tuning progress and get the model name once done
        while True:
            status = openai.fine_tunes.retrieve(self.fine_tune_id)
            status_state = status.status
            if status_state == "succeeded":
                model_name = status.fine_tuned_model
                print(f"Fine-tuning completed. Model name: {model_name}")
                break
            elif status_state == "failed":
                print("Fine-tuning failed.")
                break
            else:
                print(f"Fine-tuning status: {status_state}. Waiting for completion...")
                time.sleep(60)  # Check every 60 seconds


fineTuner = FineTuner()
fineTuner.upload_file()
fineTuner.fine_tune()
