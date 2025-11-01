import numpy as np

class SimplePerceptron:
    def __init__(self, input_size: int, random_state: int = 42):
        np.random.seed(random_state)
        self.weights = np.random.randn(input_size)
        self.bias = 0.0

    def threshold_function(self, value: float):
        return 1 if value >= 0 else 0

    def predict(self, x: np.ndarray):
        linear_product = np.dot(x, self.weights) + self.bias
        return self.threshold_function(linear_product)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.01, epochs: int = 100):
        epoch = 0
        while epoch < epochs:
            total_error = 0.0
            for x_i, y_i in zip(X, Y):
                output = np.dot(x_i, self.weights) + self.bias
                error = y_i - output
                delta_weights = learning_rate * error * x_i
                self.weights += delta_weights
                self.bias += learning_rate * error
                total_error += abs(error)
            
            print(f"Epoch: {epoch},  Error: {total_error}", flush=True)
            epoch += 1

class TwoLayersPerceptron:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, random_state: int = 42):
        np.random.seed(random_state)
        self.W1 = np.random.randn(input_size, hidden_size)    # Input to hidden
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)   # Hidden to output
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x: np.ndarray):
        return x * (1 - x)

    def predict(self, X: np.ndarray):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        output = self._sigmoid(z2)
        return (output > 0.5).astype(int)

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.01, epochs: int = 1000):
        for epoch in range(epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            output = self._sigmoid(z2)
            # Backward pass
            error = Y - output
            d_output = error * self._sigmoid_deriv(output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self._sigmoid_deriv(a1)
            # Update weights and biases
            self.W2 += a1.T.dot(d_output) * learning_rate
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            self.W1 += X.T.dot(d_hidden) * learning_rate
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

            print(f"Epoch: {epoch}, Error: {error}", flush=True)


if __name__ == "__main__":
    perceptron = SimplePerceptron(2)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 0, 0, 1])
    perceptron.fit(X, Y, epochs=10)
    print(f"Final weights: {perceptron.weights}, bias: {perceptron.bias}")

    print("Output: ", perceptron.predict(np.array([0, 0])), "Expected: 1")
    print("Output: ", perceptron.predict(np.array([0, 1])), "Expected: 0")
    print("Output: ", perceptron.predict(np.array([1, 0])), "Expected: 0")
    print("Output: ", perceptron.predict(np.array([1, 1])), "Expected: 1")
    
