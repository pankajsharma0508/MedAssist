from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from tasks.SpeechToTextConverter import SpeechToTextConverter

app = FastAPI()
speechToTextConverter = SpeechToTextConverter()


@app.get("/")
async def index():
    return "<h1> Hello Worlds</h1>"


@app.get("/speech-to-text")
async def ConvertSpeechToText(file: Annotated[UploadFile, File(...)]):
    return speechToTextConverter.convert()
