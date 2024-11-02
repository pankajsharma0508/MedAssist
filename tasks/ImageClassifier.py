from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
labels = ["Chest X-Ray", "CT-Scan", "Lab Report"]


class ImageClassifier:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def categorize(self, imageUrl):
        image = Image.open(requests.get(imageUrl, stream=True).raw)
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        top_label, top_prob = max(zip(labels, probs[0]), key=lambda x: x[1])
        return f"{top_label} ({top_prob.item() * 100:.2f}%)"
