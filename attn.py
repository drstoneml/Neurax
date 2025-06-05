import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def one_hot_encode(word, word_to_index):
    vector = np.zeros(len(word_to_index))
    vector[word_to_index[word]] = 1
    return vector

class MultiHeadAttention:
    def __init__(self, num_heads, embedding_dim, vocab_size):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.vocab_size = vocab_size

        self.Wq = np.random.randn(num_heads, embedding_dim, self.head_dim) * 0.01
        self.Wk = np.random.randn(num_heads, embedding_dim, self.head_dim) * 0.01
        self.Wv = np.random.randn(num_heads, embedding_dim, self.head_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)

    def attention(self, Q, K, V):
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(Q.shape[-1])
        scores = np.clip(scores, -50, 50)  # Prevent overflow
        weights = softmax(scores, axis=-1)
        return np.matmul(weights, V), weights
 
    def forward(self, x):
        self.x = x  # Cache input
        batch_size, seq_len, _ = x.shape
        self.Q, self.K, self.V = [], [], []
        self.head_outputs, self.attn_weights = [], []

        for h in range(self.num_heads):
            q = x @ self.Wq[h]
            k = x @ self.Wk[h]
            v = x @ self.Wv[h]
            out, w = self.attention(q, k, v)
            self.Q.append(q)
            self.K.append(k)
            self.V.append(v)
            self.attn_weights.append(w)
            self.head_outputs.append(out)

        self.concat_heads = np.concatenate(self.head_outputs, axis=-1)
        self.logits = self.concat_heads @ self.W_out + self.b_out # (embedding_dim, vocab_size)
        return self.concat_heads

    def backward(self, y_true, y_pred):
        batch_size, seq_len, vocab_size = y_pred.shape
        d_logits = softmax(y_pred) - y_true  # Cross-entropy derivative
        d_concat = d_logits @ self.W_out.T
        dW_out = self.concat_heads.reshape(-1, self.embedding_dim).T @ d_logits.reshape(-1, vocab_size)
        db_out = np.sum(d_logits, axis=(0, 1))

        # Split d_concat into heads
        d_heads = np.split(d_concat, self.num_heads, axis=-1)
        dWq, dWk, dWv = np.zeros_like(self.Wq), np.zeros_like(self.Wk), np.zeros_like(self.Wv)

        for h in range(self.num_heads):
            d_out = d_heads[h]
            V = self.V[h]
            W = self.attn_weights[h]
            dV = W.transpose(0, 2, 1) @ d_out
            dW = d_out @ V.transpose(0, 2, 1)

            dQ = dW @ self.K[h]
            dK = dW.transpose(0, 2, 1) @ self.Q[h]

            x = self.x
            dWq[h] = np.einsum('bij,bik->jk', x, dQ)
            dWk[h] = np.einsum('bij,bik->jk', x, dK)
            dWv[h] = np.einsum('bij,bik->jk', x, dV)

        return dWq, dWk, dWv, dW_out, db_out

    def update(self, dWq, dWk, dWv, dW_out, db_out, lr):
        self.Wq -= lr * dWq
        self.Wk -= lr * dWk
        self.Wv -= lr * dWv
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out
    def get_logits(self, attn_output):
        return attn_output @ self.W_out + self.b_out
    def train(self, inputs, targets, learning_rate=0.001, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                x = inputs[i:i+1]
                y_true = targets[i:i+1]
                y_pred = self.forward(x)
                y_pred = self.logits  # Use logits for loss calculation
                loss = -np.sum(y_true * np.log(softmax(y_pred) + 1e-9))
                total_loss += loss
                grads = self.backward(y_true, y_pred)
                self.update(*grads, lr=learning_rate)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(inputs):.4f}")
# === Load vocab and data ===
data = np.load('word2vec_model.npz')
W1 = data['W1']
with open('wikidata.txt', 'r') as f:
    text = f.read()
tokens = text.split()
vocab = sorted(set(tokens))
word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for i, w in enumerate(vocab)}

# === Prepare input and target data ===
# === Prepare input and target data ===
seq_length = 5
num_samples = len(tokens) - seq_length  # Use all data
inputs, targets = [], []
for i in range(num_samples):
    idxs = [word_to_index[tok] for tok in tokens[i:i+seq_length]]
    next_word = tokens[i + seq_length]
    inputs.append(W1[idxs])
    targets.append(one_hot_encode(next_word, word_to_index))
inputs = np.array(inputs)  # (batch, seq, embed)
targets = np.array(targets).reshape(num_samples, 1, -1)
targets = np.repeat(targets, seq_length, axis=1)  # (batch, seq, vocab)
model = MultiHeadAttention(num_heads=2, embedding_dim=W1.shape[1], vocab_size=len(vocab))
model.train(inputs, targets, learning_rate=0.001, epochs=10)