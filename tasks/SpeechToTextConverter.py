import os, torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pathlib import Path

# Get the current working directory


class SpeechToTextConverter:
    # Initializer / Instance Attributes
    def __init__(self):
        # load model and tokenizer
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def convert(self):
        parent_folder_path = Path.cwd().parent
        file_path = os.path.join(parent_folder_path, "audio", "harvard.wav")
        audio_input, sample_rate = torchaudio.load(file_path)
        # Resample the audio if necessary (Wav2Vec2 expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio_input = resampler(audio_input)
            # tokenize
        input_values = self.processor(
            audio_input.squeeze().numpy(), return_tensors="pt", sampling_rate=16000
        ).input_values

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]


print(SpeechToTextConverter().convert())
