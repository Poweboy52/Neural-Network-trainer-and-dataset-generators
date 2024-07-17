import numpy as np
import nnfs
import random
from nnfs.datasets import spiral_data, simple_data, sine_data, vertical_data
import atexit

nnfs.init()

def exit_handler():
    print("Optimal_loss:", optimal_loss, "Optimal accuracy:", optimal_accuracy)
    for i in range(samples):
        correct_list = []
        for ix in range(class_amount):
            correct_list.insert(0, (y[samples * ix + i], activation3.output[samples * ix + i]))
        print(correct_list)
    print("Optimal_loss:", optimal_loss, "Optimal accuracy:", optimal_accuracy)

atexit.register(exit_handler)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights, biases):
        self.weights = weights + 0.0000001
        self.biases = biases + 0.0000001
        self.weights += 0.005 * np.random.randn(n_inputs, n_neurons)
        self.biases += 0.002 * np.random.randn(n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.weights, self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#class Activation_Softmax_Single:
#    def forward(self, inputs):
#        self.output = inputs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

#class Loss_CategoricalCrossentropy_Single(Loss):
#    def forward(self, y_pred, y_true):
#
#        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
#
#        negative_log_likelihoods = -np.log(abs(y_pred_clipped - y_true))
#
#        return negative_log_likelihoods

samples = 20000
class_amount = 2
n_inputs1 = 2
weight_start_randomization = 0.01
n_neurons1 = 16
n_inputs2 = 16
n_neurons2 = 16
n_inputs3 = 16
n_neurons3 = class_amount

optimal_loss = 1000000
optimal_accuracy = 0

weights1 = weight_start_randomization * np.random.randn(n_inputs1, n_neurons1)
biases1 = np.zeros((1, n_neurons1))
weights2 = weight_start_randomization * np.random.randn(n_inputs2, n_neurons2)
biases2 = np.zeros((1, n_neurons2))
weights3 = weight_start_randomization * np.random.randn(n_inputs3, n_neurons3)
biases3 = np.zeros((1, n_neurons3))

while True:
    #X, y = spiral_data(samples=samples, classes=class_amount)
    #X, y = sine_data(samples=samples)
    #X, y = vertical_data(samples=samples, classes=class_amount)
    X, y = spiral_data(samples=samples, classes=class_amount)
    dense1 = Layer_Dense(n_inputs1, n_neurons1, weights1, biases1)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_inputs2, n_neurons2, weights2, biases2)
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(n_inputs3, n_neurons3, weights3, biases3)
    activation3 = Activation_Softmax()

    weights1, biases1 = dense1.forward(X)
    activation1.forward(dense1.output)

    weights2, biases2 = dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    weights3, biases3 = dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    for i in range(5):
        correct_list = []
        for ix in range(class_amount):
            correct_list.insert(0, y[samples * ix + i])
        print(correct_list)
    #print(X[:5])
    print(activation3.output[:5])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation3.output, y)

    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y)

    #print(activation3.output[round(samples / 3)])
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("Optimal loss:", optimal_loss)
    print("Optimal accuracy:", optimal_accuracy)
    if loss < optimal_loss:
        optimal_weights1 = weights1
        optimal_weights2 = weights2
        optimal_weights3 = weights3
        optimal_biases1 = biases1
        optimal_biases2 = biases2
        optimal_biases3 = biases3
        optimal_loss = loss
        optimal_accuracy = accuracy
    weights1 = optimal_weights1
    weights2 = optimal_weights2
    weights3 = optimal_weights3
    biases1 = optimal_biases1
    biases2 = optimal_biases2
    biases3 = optimal_biases3
