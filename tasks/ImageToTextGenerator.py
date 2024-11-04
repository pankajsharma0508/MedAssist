from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import os
from pathlib import Path


# load image from the IAM database (actually this model is meant to be used on printed text)
# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
url = "https://www.researchgate.net/profile/Lynda-Knobeloch/publication/15643132/figure/tbl1/AS:601606458454063@1520445565840/Medical-test-results-and-personal-data.png"

parent_folder_path = Path.cwd().parent
file_path = os.path.join(parent_folder_path, "reports", "report.jpg")
audio_file = open(file_path, "rb")
image = Image.open(audio_file).convert("RGB")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
