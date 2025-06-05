import numpy as np
from attn import MultiHeadAttention, model
from mlp import LinearLayer

# --- Load vocab and embeddings ---
with open("wikidata.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokens = text.split()
vocab = sorted(set(tokens))
vocab_size = len(vocab)
seq_length = 5

W = np.load("word2vec_model.npz")
W1 = W['W1']  # shape: (vocab_size, embedding_dim)
embedding_dim = W1.shape[1]

word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

def tokens_to_indices(tokens, word_to_index):
    idxs = [word_to_index.get(tok, 0) for tok in tokens]
    return [min(max(i, 0), W1.shape[0] - 1) for i in idxs]

def softmax(x, axis=-1):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -50, 50)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def clip_gradients(grad, clip_value=1.0):
    return np.clip(grad, -clip_value, clip_value)

# --- Transformer block ---
class TransformerBlock:
    def __init__(self, embedding_dim, num_heads, ff_dim, vocab_size, Attn):
        self.attn = Attn
        self.ff = LinearLayer(embedding_dim, ff_dim, embedding_dim)
        self.out = LinearLayer(embedding_dim, ff_dim, vocab_size)
    def forward(self, x):
        attn_out = self.attn.forward(x)  # (batch, seq, embedding_dim)
        x = x + attn_out
        ff_out = self.ff.forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        x = x + ff_out
        logits = self.attn.get_logits(x)  # (batch, seq, vocab_size)
        return logits

    def train(self, inputs, targets, epochs=5, lr=0.001, batch_size=16):
        num_samples = inputs.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            total_loss = 0.0
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                x_batch = inputs[idx]
                y_batch = targets[idx]

                # Forward pass
                attn_out = self.attn.forward(x_batch)
                x_ff = x_batch + attn_out
                ff_out = self.ff.forward(x_ff.reshape(-1, x_ff.shape[-1])).reshape(x_ff.shape)
                x_out = x_ff + ff_out
                logits = self.out.forward(x_out.reshape(-1, x_out.shape[-1])).reshape(x_out.shape[0], x_out.shape[1], -1)
                logits_last = logits[:, -1, :]

                probs = softmax(logits_last)
                if np.isnan(probs).any():
                    raise ValueError("NaN in probabilities")

                loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-9), axis=-1))
                d_logits = probs - y_batch
                d_x_out, dW_out, db_out = self.out.backward(x_out[:, -1, :], d_logits)
                if np.isnan(loss):
                    raise ValueError(   "NaN in loss")

                total_loss += loss * len(idx)

                d_x_ff = np.zeros_like(x_ff)
                d_x_ff[:, -1, :] = d_x_out
                d_x_ff = d_x_ff.reshape(-1, d_x_ff.shape[-1])
                x_ff_flat = x_ff.reshape(-1, x_ff.shape[-1])
                d_x_ff, dW_ff, db_ff = self.ff.backward(x_ff_flat, d_x_ff)

                # Clip & update
                self.out.update(clip_gradients(dW_out), clip_gradients(db_out), lr)
                self.ff.update(clip_gradients(dW_ff), clip_gradients(db_ff), lr)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_samples:.6f}")

# --- Prepare training data ---
def prepare_training_data(tokens, seq_length, word_to_index, W1, vocab_size):
    inputs, targets = [], []
    for i in range(len(tokens) - seq_length):
        idxs = tokens_to_indices(tokens[i:i+seq_length], word_to_index)
        next_word = tokens[i + seq_length]
        next_idx = word_to_index.get(next_word, 0)
        inputs.append(W1[idxs])
        one_hot = np.zeros(vocab_size)
        one_hot[next_idx] = 1
        targets.append(one_hot)
    return np.array(inputs), np.array(targets)

# --- Text generation ---
def generate_text(prompt, max_new_tokens=20):
    prompt_tokens = prompt.strip().split()
    generated = prompt_tokens.copy()
    for _ in range(max_new_tokens):
        input_idxs = tokens_to_indices(generated[-seq_length:], word_to_index)
        if len(input_idxs) < seq_length:
            input_idxs = [0] * (seq_length - len(input_idxs)) + input_idxs
        input_embeds = W1[input_idxs]
        input_embeds = input_embeds[np.newaxis, ...]
        logits = transformer.forward(input_embeds)
        probs = softmax(logits[0, -1, :])
        probs = probs / np.sum(probs)  # Ensure it's a valid probability distribution
        next_idx = int(np.random.choice(len(probs), p=probs))
        next_word = index_to_word[next_idx]
        generated.append(next_word)
    return " ".join(generated)

# --- Run ---
if __name__ == "__main__":
    transformer = TransformerBlock(embedding_dim=embedding_dim, num_heads=2, ff_dim=64, vocab_size=vocab_size, Attn=model)
    inputs, targets = prepare_training_data(tokens, seq_length, word_to_index, W1, vocab_size)
    transformer.train(inputs, targets, epochs=5, lr=0.01, batch_size=16)

    prompt = input("Enter prompt: ")
    output = generate_text(prompt, max_new_tokens=20)
    print("Generated:", output)
