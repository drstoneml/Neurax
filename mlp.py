import numpy as np

class LinearLayer:
    def __init__(self, input_size, hidden_size, output_size):
        # Single linear layer: ignore hidden_size for compatibility
        self.W = np.random.randn(input_size, output_size).astype(np.float32)
        self.b = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        """
        Forward pass through the linear layer.
        Args:
            x: Input data of shape (batch_size, input_size).
        Returns:
            Output of shape (batch_size, output_size).
        """
        return np.dot(x, self.W) + self.b

    def backward(self, x, grad_output):
        """
        Backward pass through the linear layer.
        Args:
            x: Input data of shape (batch_size, input_size).
            grad_output: Gradient of the loss with respect to the output.
        Returns:
            grad_input: Gradient of the loss with respect to the input.
            grad_W: Gradient of the loss with respect to the weights.
            grad_b: Gradient of the loss with respect to the biases.
        """
        grad_W = np.dot(x.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input, grad_W, grad_b

    def update(self, grad_W, grad_b, learning_rate):
        """
        Update the weights and biases using the gradients.
        Args:
            grad_W: Gradient of the loss with respect to the weights.
            grad_b: Gradient of the loss with respect to the biases.
            learning_rate: Learning rate for the update.
        """
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

    def train(self, x, y_true, learning_rate=0.001, epochs=10):
        """
        Train the linear layer.
        Args:
            x: Input data of shape (batch_size, input_size).
            y_true: True labels of shape (batch_size, output_size).
            learning_rate: Learning rate for the update.
            epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(x)
            # Compute loss (mean squared error)
            loss = np.mean((y_pred - y_true) ** 2)
            # Backward pass
            grad_output = y_pred - y_true
            grad_input, grad_W, grad_b = self.backward(x, grad_output)
            # Update weights and biases
            self.update(grad_W, grad_b, learning_rate)
            # Print loss
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")

# Example usage
if __name__ == "__main__":
    # Create a linear layer with input size 10 and output size 2
    layer = LinearLayer(input_size=10, hidden_size=5, output_size=2)
    x = np.random.rand(100, 10).astype(np.float32)
    y_true = np.random.rand(100, 2).astype(np.float32)
    layer.train(x, y_true, learning_rate=0.01, epochs=50)