import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import warnings

# Suppress FutureWarning regarding torch.load weights_only parameter.
warnings.filterwarnings("ignore", category=FutureWarning)

#############################
# Custom Tokenizer Function #
#############################
def tokenizer(sentences, max_length=128, padding='max_length', truncation=True):
    tokenized_outputs = {"input_ids": [], "attention_mask": []}
    for sentence in sentences:
        tokens = sentence.lower().split()
        # Use [UNK] if token not found (default id is 4)
        token_ids = [word2id.get(token, word2id.get('[UNK]', 4)) for token in tokens]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        if padding == 'max_length':
            pad_len = max_length - len(token_ids)
            token_ids += [word2id.get('[PAD]', 0)] * pad_len
            attention_mask += [0] * pad_len
        tokenized_outputs["input_ids"].append(token_ids)
        tokenized_outputs["attention_mask"].append(attention_mask)
    return tokenized_outputs

####################
# Mean Pooling     #
####################
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

#########################
# BERT Model Components #
#########################
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, seq_k.size(1))

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = nn.Linear(self.n_heads * self.d_k, self.d_model).to(self.device)(context)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

####################################
# BERT Model (with Pretraining Ops)
####################################
class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # Shared decoder with embedding weights
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        h_pooled = self.activ(self.fc(output[:, 0]))  # CLS token representation
        logits_nsp = self.classifier(h_pooled)
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp

    # Helper for sentence encoding (for inference)
    def get_last_hidden_state(self, input_ids, attention_mask):
        segment_ids = torch.zeros_like(input_ids).to(self.device)
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output

#############################################
# Functions to Load the Model and Tokenizer #
#############################################
def load_model_and_tokenizer(device):
    checkpoint_path = os.path.join("Models", "final_model.pth")
    # Try loading with weights_only=True if supported
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    global word2id
    word2id = checkpoint["word2id"]  # checkpoint must include word2id
    vocab_size = checkpoint["vocab_size"]
    
    # Set hyperparameters (must match training)
    n_layers = 12
    n_heads = 12
    d_model = 768
    d_ff = d_model * 4
    d_k = 64
    n_segments = 2
    max_len = 1000

    model = BERT(n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create classifier head instance and load state from checkpoint
    classifier_head = torch.nn.Linear(d_model * 3, 3).to(device)
    classifier_head.load_state_dict(checkpoint["classifier_head_state_dict"])
    classifier_head.eval()
    
    return model, classifier_head, tokenizer, word2id

###########################################
# NLI Prediction Function for the Web App #
###########################################
def predict_nli(model, classifier_head, sentence_a, sentence_b, device, max_length=128):
    # Tokenize sentences using the custom tokenizer
    inputs_a = tokenizer([sentence_a], max_length=max_length, padding="max_length", truncation=True)
    inputs_b = tokenizer([sentence_b], max_length=max_length, padding="max_length", truncation=True)
    
    # Convert lists to torch tensors and move to device
    input_ids_a = torch.tensor(inputs_a["input_ids"]).to(device)
    attention_a = torch.tensor(inputs_a["attention_mask"]).to(device)
    input_ids_b = torch.tensor(inputs_b["input_ids"]).to(device)
    attention_b = torch.tensor(inputs_b["attention_mask"]).to(device)
    
    # Obtain token embeddings using the helper method
    emb_a = model.get_last_hidden_state(input_ids_a, attention_a)
    emb_b = model.get_last_hidden_state(input_ids_b, attention_b)
    
    # Mean-pool the token embeddings to get sentence-level embeddings
    mean_a = mean_pool(emb_a, attention_a)  # shape: [1, hidden_dim]
    mean_b = mean_pool(emb_b, attention_b)  # shape: [1, hidden_dim]
    
    # Compute the absolute difference between embeddings
    diff = torch.abs(mean_a - mean_b)
    # Concatenate features: mean_a, mean_b, diff (resulting shape: [1, hidden_dim*3])
    features = torch.cat([mean_a, mean_b, diff], dim=1)
    
    # Get logits from the classifier head and predict the NLI label
    logits = classifier_head(features)
    predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    
    # Calculate cosine similarity between the mean-pooled embeddings
    cosine_sim = (mean_a * mean_b).sum(dim=1) / (torch.norm(mean_a, dim=1) * torch.norm(mean_b, dim=1) + 1e-8)
    similarity_score = cosine_sim.item()
    
    return label_map[predicted_class], similarity_score
