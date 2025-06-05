import dpnp as np
import numpy as cpu_np  # Import CPU-based numpy for shuffling

# Load and preprocess the text
with open('wikidata.txt', 'r') as f:
    text = f.read()

# Split text into sentences and tokens
sentences = text.split('.')
tokens = text.split()
vocab = sorted(set(tokens))  # Sorted vocabulary for consistent indexing
word_to_index = {word: i for i, word in enumerate(vocab)}  # Map words to indices
index_to_word = {i: word for i, word in enumerate(vocab)}  # Map indices to words

# Function to generate training data
def get_training_data(sentences, window_size=1):
    training_data = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if word not in word_to_index:
                continue  # Skip words not in the vocabulary
            context_indices = list(range(max(0, i - window_size), min(len(words), i + window_size + 1)))
            context_indices.remove(i)  # Remove the target word index
            for context_index in context_indices:
                context_word = words[context_index]
                if context_word in word_to_index:
                    training_data.append((word_to_index[word], word_to_index[context_word]))
    return np.array(training_data, dtype=np.int32)  # Convert to numpy array with int32 type

# Generate training data
training_data = get_training_data(sentences)

# Softmax function
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    x = x.astype(np.float32)  # Ensure input is float32
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)

# CBOW Word2Vec Model
class CBOW_Word2vec:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize weights with random values and cast to float32
        self.W1 = np.random.random((vocab_size, embedding_dim)).astype(np.float32)  # Input weights
        self.W2 = np.random.random((embedding_dim, vocab_size)).astype(np.float32)  # Output weights

    def train(self, training_data, learning_rate=0.01, epochs=50, batch_size=256):
        num_batches = len(training_data) // batch_size
        for epoch in range(epochs):
            cpu_np.random.shuffle(training_data)  # Shuffle data at the start of each epoch
            total_loss = 0

            for batch in range(num_batches):
                batch_data = training_data[batch * batch_size:(batch + 1) * batch_size]
                x = np.zeros((batch_size, self.vocab_size), dtype=np.float32)
                y_true = np.zeros((batch_size, self.vocab_size), dtype=np.float32)

                for i, (word, context_word) in enumerate(batch_data):
                    x[i, word] = 1
                    y_true[i, context_word] = 1

                # Forward pass
                h = np.dot(x, self.W1)  # Hidden layer
                y_pred = softmax(np.dot(h, self.W2))  # Output layer

                # Compute loss (cross-entropy)
                loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / batch_size
                total_loss += loss

                # Backward pass (gradient descent)
                e = y_pred - y_true  # Error term
                dW2 = np.dot(h.T, e) / batch_size  # Gradient for W2
                dW1 = np.dot(x.T, np.dot(e, self.W2.T)) / batch_size  # Gradient for W1

                # Update weights
                self.W1 -= learning_rate * dW1
                self.W2 -= learning_rate * dW2

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss}')

    def get_word_vector(self, word_index):
        """Get the word vector for a given word index."""
        return self.W1[word_index]

    def get_similar_words(self, word_index, top_n=5):
        """Get the top N most similar words to a given word index."""
        word_vector = self.get_word_vector(word_index)
        similarities = np.dot(self.W1, word_vector)
        similar_indices = np.argsort(similarities)[-top_n:][::-1]
        return similar_indices, similarities[similar_indices]
    def save_model(self, filename):
        """Save the model to a file."""
        import numpy as cpu_np  # Use CPU-based numpy for saving
        # Convert dpnp arrays to numpy arrays before saving
        W1_numpy = self.W1.asnumpy()
        W2_numpy = self.W2.asnumpy()
        cpu_np.savez(filename, W1=W1_numpy, W2=W2_numpy)

    def load_model(self, filename):
        """Load the model from a file."""
        import numpy as cpu_np  # Use CPU-based numpy for loading
        data = cpu_np.load(filename)
        # Convert numpy arrays back to dpnp arrays
        self.W1 = np.array(data['W1'])
        self.W2 = np.array(data['W2'])

    

# Initialize and train the CBOW model
print(len(vocab))
model = CBOW_Word2vec(vocab_size=len(vocab), embedding_dim=100)
model.train(training_data, learning_rate=0.05, epochs=30, batch_size=512)

# Save the trained model
model.save_model('word2vec_model.npz')

# Example: Get similar words
word_index = word_to_index['example'] if 'example' in word_to_index else 0  # Replace 'example' with a word in your vocab
similar_indices, similarities = model.get_similar_words(word_index, top_n=5)