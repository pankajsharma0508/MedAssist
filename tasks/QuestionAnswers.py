import openai
import os

openai.api_key = os.getenv("OPEN_AI_KEY")


class QuestionAnswers:
    def __init__(self):
        self.model = "ft:gpt-3.5-turbo-0125:personal::AQkdF1Gb"

    def get_answer_for_question(self, question):
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant.",
                },
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content


# question = "I may have itching vomiting fatigue weight_loss high_fever yellowish_skin dark_urine."
# print(QuestionAnswers().get_answer_for_question(question))
