from flask import Flask, render_template, request
from util_funs import load_model_and_tokenizer, calculate_similarity
import torch

app = Flask(__name__)

# Load model, tokenizer, and vocabulary once on startup.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer_func, word2id = load_model_and_tokenizer(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = None
    sentence_a = ""
    sentence_b = ""
    if request.method == 'POST':
        sentence_a = request.form.get('sentence_a')
        sentence_b = request.form.get('sentence_b')
        if sentence_a and sentence_b:
            similarity = calculate_similarity(model, tokenizer_func, sentence_a, sentence_b, device)
    return render_template('index.html', similarity=similarity, sentence_a=sentence_a, sentence_b=sentence_b)

if __name__ == '__main__':
    app.run(debug=True)
