import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load models and tokenizer from Hugging Face
models = {
    # "Odd-Layer": AutoModelForSequenceClassification.from_pretrained("your_username/Odd-Layer_toxic_comment"),
    # "Even-Layer": AutoModelForSequenceClassification.from_pretrained("your_username/Even-Layer_toxic_comment"),
    "LoRA": AutoModelForSequenceClassification.from_pretrained("st125338/lora_toxic_comment")
}
tokenizer = AutoTokenizer.from_pretrained("st125338/lora_toxic_comment")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models.values():
    model.to(device)

# Prediction function
def predict(text, model_name):
    model = models[model_name]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()
    return "Toxic" if pred == 1 else "Non-Toxic"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text here..."),
        gr.Dropdown(choices=["LoRA"], label="Select Model")
    ],
    outputs="text",
    title="Toxic Comment Classifier",
    description="Select a model and enter text to classify as Toxic or Non-Toxic."
)

iface.launch()