from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

app = FastAPI()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load ONNX model
session = ort.InferenceSession("model.onnx")

@app.get("/")
def home():
    return {"message": "ONNX FastAPI Project Running 🚀"}

@app.post("/analyze")
def analyze_text(text: str):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="np")

    # Run ONNX model
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
    )

    # Get embeddings
    vector = outputs[0]

    # 👉 Convert to simple readable result
    mean_value = float(np.mean(vector))

    # Simple demo logic
    if mean_value > 0:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😐"

    return {
        "input_text": text,
        "analysis": sentiment,
        "score": round(mean_value, 4)
    }