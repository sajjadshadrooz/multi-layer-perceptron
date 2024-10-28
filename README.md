# multi-layer-perceptron
<img src="https://www.go-rbcs.com/wp-content/uploads/2019/06/simple-and-deep-neural-networks.png" />
<br/>
MLP stands for Multilayer Perceptron, which is a type of artificial neural network. 
<hr/>
Here are some key characteristics of MLPs:
Architecture: MLPs consist of multiple layers of nodes (neurons):
<br/>
<ul>
  <li>Input Layer: Receives the input data.</li>
  <li>Hidden Layers: One or more layers where the computations take place. Each neuron in a hidden layer applies a weighted sum of its inputs, followed by a nonlinear activation function.</li>
  <li>Output Layer: Produces the final output, which can be for regression or classification tasks.</li>
  <li>Feedforward Network: Information moves in one direction—from the input layer through the hidden layers to the output layer. There are no cycles or loops.</li>
</ul>

> [!NOTE]
> At this project we preprocess dataset, numeric columns come in range 0 to 1 and categorical columns come in range 0 to 1
<br/>
Activation Functions: Common activation functions include the sigmoid, tanh, and ReLU (Rectified Linear Unit). These functions introduce non-linearity into the model, allowing it to learn complex patterns.
<br/>
Backpropagation: MLPs use backpropagation for training, a method that adjusts the weights of the network based on the error of the output compared to the desired output. This process is typically done using gradient descent optimization techniques.
<br/>
Applications: MLPs are widely used in various applications, including image recognition, natural language processing, and time-series prediction, due to their ability to approximate complex functions.
<br/>
MLPs are a foundational concept in deep learning and serve as the basis for more advanced neural network architectures.

