from flask import Flask, render_template, request
import torch
from torch import nn
import math, datasets, torchtext, pickle

app = Flask(__name__)

# Load your trained LSTM model
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# Load the vocabulary
with open('../model/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load model
vocab_size = len(vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
model.load_state_dict(torch.load('../model/st125338_best-val-lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()

# Tokenizer
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Text generation function
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device='cpu', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()
            if prediction == vocab['<eos>']:
                break
            indices.append(prediction)
    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return ' '.join(tokens)

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        if len(prompt.split()) > 5:
            return "Please enter no more than 5 words."
        generated_text = generate(prompt, max_seq_len=50, temperature=1.0, model=model, tokenizer=tokenizer, vocab=vocab)
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    return render_template('index.html', prompt=None, generated_text=None)

if __name__ == '__main__':
    app.run(debug=True)