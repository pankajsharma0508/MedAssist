import openai
import os
from pathlib import Path


class SpeechToTextConverter:
    def __init__(self):
        openai.api_key = os.getenv("OPEN_AI_KEY")

    async def convert(self, audio_file):
        parent_folder_path = Path.cwd().parent
        file_path = os.path.join(
            parent_folder_path, "MEDASSIST", "audio", "diagnosis.wav"
        )

        with open(file_path, "wb") as f:
            f.write(await audio_file.read())

        audio_file = open(file_path, "rb")
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        print(transcription.text)
        return transcription.text
