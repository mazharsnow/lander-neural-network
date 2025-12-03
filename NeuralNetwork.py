# NeuralNetwork.py
import math
import random
import json

# ---------- Activation ----------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)


# ---------- Neuron ----------
class Neuron:
    def __init__(self, n_inputs: int):
        # Random initial weights and bias
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.bias = random.uniform(-1.0, 1.0)
        self.output = 0.0
        self.delta = 0.0

    def activate(self, inputs):
        z = self.bias
        for w, x in zip(self.weights, inputs):
            z += w * x
        self.output = sigmoid(z)
        return self.output


# ---------- Simple 1-hidden-layer MLP ----------
class MLP:
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.hidden_layer = [Neuron(n_inputs) for _ in range(n_hidden)]
        self.output_layer = [Neuron(n_hidden) for _ in range(n_outputs)]

    def forward(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return outputs

    def backward(self, expected):
        # Output layer deltas
        for i, neuron in enumerate(self.output_layer):
            error = expected[i] - neuron.output
            neuron.delta = error * sigmoid_derivative(neuron.output)

        # Hidden layer deltas
        for j, hidden_neuron in enumerate(self.hidden_layer):
            error = 0.0
            for output_neuron in self.output_layer:
                error += output_neuron.weights[j] * output_neuron.delta
            hidden_neuron.delta = error * sigmoid_derivative(hidden_neuron.output)

    def update_weights(self, inputs, learning_rate):
        # Update hidden layer
        for neuron in self.hidden_layer:
            for j in range(len(inputs)):
                neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
            neuron.bias += learning_rate * neuron.delta

        # Update output layer
        hidden_outputs = [n.output for n in self.hidden_layer]
        for neuron in self.output_layer:
            for j in range(len(hidden_outputs)):
                neuron.weights[j] += learning_rate * neuron.delta * hidden_outputs[j]
            neuron.bias += learning_rate * neuron.delta

    def train(self, dataset, n_epochs=100, learning_rate=0.1, verbose=True):
        n_samples = len(dataset)
        for epoch in range(n_epochs):
            sum_sq_error = 0.0
            for inputs, targets in dataset:
                outputs = self.forward(inputs)
                for o, t in zip(outputs, targets):
                    sum_sq_error += (t - o) ** 2
                self.backward(targets)
                self.update_weights(inputs, learning_rate)
            rmse = math.sqrt(sum_sq_error / (n_samples * self.n_outputs))
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.6f}")

    def predict(self, inputs):
        return self.forward(inputs)

    # ----- Save / load -----
    def save(self, filename):
        data = {
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "n_outputs": self.n_outputs,
            "hidden_layer": [
                {"weights": n.weights, "bias": n.bias}
                for n in self.hidden_layer
            ],
            "output_layer": [
                {"weights": n.weights, "bias": n.bias}
                for n in self.output_layer
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        mlp = MLP(data["n_inputs"], data["n_hidden"], data["n_outputs"])
        for neuron, saved in zip(mlp.hidden_layer, data["hidden_layer"]):
            neuron.weights = saved["weights"]
            neuron.bias = saved["bias"]
        for neuron, saved in zip(mlp.output_layer, data["output_layer"]):
            neuron.weights = saved["weights"]
            neuron.bias = saved["bias"]
        return mlp
