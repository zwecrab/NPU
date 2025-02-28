import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
# Load your model and tokenizer from your Hugging Face repo
repo_id = "st125338/npu_a5_dpo_qwen2_model"
model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt: str, max_length: int = 250) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # if full_text.startswith(prompt):
    #     response = full_text[len(prompt):].strip()
    # else:
    #     response = full_text.strip()
    # return response
    return full_text

# Create a Gradio Interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, placeholder="Enter your prompt here...", label="Input Prompt"),
    outputs=gr.Textbox(label="Model Response"),
    title="A5 DPO: Qwen2.0 Fine-Tuned Model",
    description="Enter a prompt and click 'Submit' to generate a response using the fine-tuned Qwen2.5 model.",
    article="<p style='text-align: center; margin-top: 20px;'>st125338 : Zwe Htet</p>"
)

iface.launch()