import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt

# Load models and tokenizers
models = {
    "Odd-Layer": {
        "model": AutoModelForSequenceClassification.from_pretrained("st125338/odd-layer-toxic-comments"),
        "tokenizer": AutoTokenizer.from_pretrained("st125338/odd-layer-toxic-comments")
    },
    "Even-Layer": {
        "model": AutoModelForSequenceClassification.from_pretrained("st125338/even-layer-toxic-comments"),
        "tokenizer": AutoTokenizer.from_pretrained("st125338/even-layer-toxic-comments")
    },
    "LoRA": {
        "model": AutoModelForSequenceClassification.from_pretrained("st125338/lora_toxic_comment"),
        "tokenizer": AutoTokenizer.from_pretrained("st125338/lora_toxic_comment")
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_info in models.values():
    model_info["model"].to(device)
    model_info["model"].eval()

def create_plot(prob_not_toxic, prob_toxic):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(['Non-Toxic', 'Toxic'], 
                 [prob_not_toxic, prob_toxic], 
                 color=['#228B22', '#DC143C'],
                 height=0.6)
    
    ax.set_xlabel('Probability', fontsize=14, color='black')
    ax.set_title('Toxicity Prediction', fontsize=16, color='black', pad=20)
    ax.tick_params(axis='both', colors='black')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', 
                va='center', 
                fontsize=14,
                color='black')
    
    plt.tight_layout()
    return fig

def predict(model_name, text):
    model_info = models[model_name]
    inputs = model_info["tokenizer"](
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model_info["model"](**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return create_plot(float(probs[0][0]), float(probs[0][1]))

description = """
**Automated Toxicity Detection in Comments**
This application utilizes state-of-the-art machine learning models to assess the toxicity of user-generated comments. By selecting a model and inputting a comment, users can obtain a probability score indicating the likelihood of the comment being toxic or non-toxic.
**How to Use:**
1. **Select a Model:** Choose from the available models in the dropdown menu
2. **Enter a Comment:** Type or paste the comment you wish to analyze
3. **View Results:** The application will display a bar chart showing the probabilities
**Interpretation:**
- **Non-Toxic:** Likely respectful and appropriate
- **Toxic:** May contain offensive or harmful content
**Note:** Example comments are for demonstration only
"""

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(list(models.keys())),  # Fixed missing parenthesis
        gr.Textbox(placeholder="Type your comment here...")
    ],
    outputs=gr.Plot(),
    title="NPU Assignment A7: Distillation, Get Smaller, Get Faster",
    description=description,
    examples=[
        ["LoRA", "You're a stupid idiot and should die!"],
        ["Odd-Layer", "This is a perfectly reasonable comment"],
        ["Even-Layer", "Go back to your country you foreigner"]
    ],
    theme=gr.themes.Default(primary_hue="blue")
)

demo.launch()