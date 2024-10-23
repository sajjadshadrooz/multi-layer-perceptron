import numpy as np
import pandas as pd

class MLP:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers) - 1 
        
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            weight = np.random.randn(layers[i], layers[i+1]) * 0.01
            bias = np.zeros((1, layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def cross_entropy_loss(self, Y_hat, Y):
        n_samples = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(n_samples), Y])
        loss = np.sum(log_likelihood) / n_samples
        return loss

    def forward_propagation(self, X):
        A = X
        activations = [A]  
        
        for i in range(self.num_layers - 1): 
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.sigmoid(Z)
            activations.append(A)

        # Output layer (use softmax)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.softmax(Z)
        activations.append(A)
        
        return activations

    def backward_propagation(self, X, Y, activations):
        
        n_samples = X.shape[0]
        Y_encoded = np.zeros((n_samples, self.layers[-1]))
        Y_encoded[range(n_samples), Y] = 1
        
        A2 = activations[-1]
        dZ = A2 - Y_encoded
        gradients_w = []
        gradients_b = []
        
        for i in reversed(range(self.num_layers)):
            A_prev = activations[i]
            dW = np.dot(A_prev.T, dZ) / n_samples
            db = np.sum(dZ, axis=0, keepdims=True) / n_samples
            
            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            
            if i > 0: 
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.sigmoid_derivative(activations[i])

        
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            activations = self.forward_propagation(X)
            loss = self.cross_entropy_loss(activations[-1], Y)
            self.backward_propagation(X, Y, activations)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)