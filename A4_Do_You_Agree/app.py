from flask import Flask, render_template, request
from util_funs import load_model_and_tokenizer, predict_nli
import torch

app = Flask(__name__)

# Load model, classifier, etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classifier_head, tokenizer, word2id = load_model_and_tokenizer(device)

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    similarity = None
    premise = ""
    hypothesis = ""
    
    if request.method == "POST":
        premise = request.form.get("premise")
        hypothesis = request.form.get("hypothesis")
        
        if premise and hypothesis:
            # predict_nli now returns (label, similarity)
            label, similarity = predict_nli(model, classifier_head, premise, hypothesis, device)
    
    # Pass label and similarity to the template
    return render_template("index.html",
                           label=label,
                           similarity=similarity,
                           premise=premise,
                           hypothesis=hypothesis)

if __name__ == "__main__":
    app.run(debug=True)
