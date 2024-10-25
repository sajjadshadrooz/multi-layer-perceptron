from .activation_functions import *

import numpy as np
import pandas as pd


class MLP:
    def __init__(self, layers, learning_rate, activation_function):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(1, len(layers)):
            weight_matrix = np.random.randn(layers[i-1], layers[i]) * 0.1
            bias_vector = np.zeros((1, layers[i]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    
    def apply_activation_function(self, x):
        match self.activation_function:
            case 'sigmoid':
                return sigmoid(x)
            case 'relu':
                return relu(x)
            
    def apply_activation_function_derivative(self, x):
        match self.activation_function:
            case 'sigmoid':
                return sigmoid_derivative(x)
            case 'relu':
                return relu_derivative(x)

    def forward(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            z_values.append(z)
            activation = self.apply_activation_function(z)
            activations.append(activation)
        
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        activation = sigmoid(z)
        activations.append(activation)
            
        return activations, z_values

    def backward(self, y, activations, z_values):
        deltas = []
        grad_weights = []
        grad_biases = []
        
        error = activations[-1] - y
        delta = error * sigmoid(activations[-1])
        deltas.append(delta)
        
        for i in range(len(self.weights) - 2, -1, -1):
            if i == len(self.weights) - 2:
                delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.apply_activation_function_derivative(z_values[i])
            else:
                delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.apply_activation_function_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()
        
        for i in range(len(self.weights)):
            grad_weight = np.dot(activations[i].T, deltas[i])
            grad_bias = np.array(np.sum(deltas[i], axis=0)).reshape(1, -1)
            grad_weights.append(grad_weight)
            grad_biases.append(grad_bias)
        
        return grad_weights, grad_biases

    def update_weights(self, grad_weights, grad_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations, z_values = self.forward(X)
            grad_weights, grad_biases = self.backward(y, activations, z_values)
            self.update_weights(grad_weights, grad_biases)
            
            if epoch % 100 == 0:
                loss = np.mean((y - activations[-1]) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.where(activations[-1] >= 0.546, 1, 0)