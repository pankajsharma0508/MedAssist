from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def index():
    return "<h1> Hello Worlds</h1>"


@app.get("/speech-to-text")
async def ConvertSpeechToText():
    return "<h1> speech-to-text</h1>"
