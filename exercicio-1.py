import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else -1

    def predict(self, inputs):
        sum_inputs = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(sum_inputs)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

X_and = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_and = np.array([1, -1, -1, -1])

X_or = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
y_or = np.array([1, 1, 1, -1])

perceptron_and = Perceptron(num_inputs=2)
perceptron_and.train(X_and, y_and)

perceptron_or = Perceptron(num_inputs=2)
perceptron_or.train(X_or, y_or)

def test_perceptron(perceptron, X, operation):
    print(f"\nResultados para {operation}:")
    for inputs in X:
        prediction = perceptron.predict(inputs)
        print(f"Entradas: {inputs}, Saída: {prediction}")

test_perceptron(perceptron_and, X_and, "AND")
test_perceptron(perceptron_or, X_or, "OR")