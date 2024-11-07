from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from tasks.SpeechToTextConverter import SpeechToTextConverter
from tasks.SummaryGenerator import SummaryGenerator
from tasks.SeverityPredictor import SeverityPredictor
from tasks.ImageClassifier import ImageClassifier
from tasks.QuestionAnswers import QuestionAnswers

from starlette.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
origins = ["http://localhost:4200"]

speechToTextConverter = SpeechToTextConverter()
summaryGenerator = SummaryGenerator()
severityPredictor = SeverityPredictor()
imageClassifier = ImageClassifier()
qAndAGenerator = QuestionAnswers()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that are allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def index():
    return "<h1> Hello Worlds</h1>"


@app.post("/speech-to-text")
async def ConvertSpeechToText(file: Annotated[UploadFile, File(...)]):
    return await speechToTextConverter.convert(file)


@app.get("/patient-severity")
async def get_patient_severity(symptoms: str):
    print(symptoms)
    return severityPredictor.predict(symptoms)


@app.get("/summarize")
async def get_summary(details: str):
    print(details)
    return summaryGenerator.summarize(details)


@app.get("/image-category")
async def get_image_category(imageUrl: str):
    print(imageUrl)
    return imageClassifier.categorize(imageUrl)


@app.get("/predict-disease")
async def predict_disease_for_symptoms(symptoms: str):
    print(symptoms)
    return qAndAGenerator.get_answer_for_question(symptoms)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
