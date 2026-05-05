from transformers import AutoTokenizer, AutoModel
import torch

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

inputs = tokenizer("This is a sample text", return_tensors="pt")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=18
)

print("✅ ONNX model created successfully!")