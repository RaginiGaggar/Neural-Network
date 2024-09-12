# Neural Network from Scratch 

## Overview

This repository contains a neural network implemented from scratch to classify handwritten digits using the MNIST dataset. The network is built using basic neural network components, including fully connected layers and various activation functions, and trained using gradient descent.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Network Architecture](#network-architecture)
- [Activation Functions](#activation-functions)
- [Loss Function](#loss-function)
- [Training](#training)
- [Testing and Results](#testing-and-results)
- [Usage](#usage)

## Introduction

This project demonstrates how to build a neural network from scratch using NumPy. The neural network is designed to classify digits from the MNIST dataset, a popular dataset for handwritten digit recognition.

## Data Preprocessing

The MNIST dataset is loaded and preprocessed as follows:

1. **Flattening**: Each 28x28 image is flattened into a 784x1 vector to convert the 2D image data into a 1D array suitable for processing by the neural network.
2. **Normalization**: Pixel values are normalized to the range [0, 1] by dividing by 255. This ensures that the input features are on a similar scale, which helps the training process converge more effectively.
3. **One-Hot Encoding**: Labels are converted to one-hot encoded vectors, where each class is represented by a vector with a single 1 and the rest 0. This representation is suitable for classification tasks.

### Code Example

```python
def preprocess_data(x, y, limit=None):
    x = x.reshape(x.shape[0], 784, 1)
    x = x.astype("float32") / 255
    y = to_categorical(y, 10)
    y = y.reshape(y.shape[0], 10, 1)
    if limit:
        return x[:limit], y[:limit]
    return x, y
```

## Network Architecture

The neural network consists of the following layers:

1. **Dense Layer (784 → 128)**: A fully connected layer with 128 neurons, transforming the 784-dimensional input into a 128-dimensional space.
2. **Activation (Tanh)**: The hyperbolic tangent activation function, which introduces non-linearity to the model.
3. **Dense Layer (128 → 64)**: A fully connected layer with 64 neurons, further transforming the 128-dimensional output into a 64-dimensional space.
4. **Activation (Tanh)**: Another hyperbolic tangent activation function for further non-linearity.
5. **Dense Layer (64 → 10)**: The final fully connected layer with 10 neurons, representing the 10 possible digit classes.
6. **Softmax Activation**: Converts the output into a probability distribution over the 10 classes.

### Code Example

```python
network = [
    Dense(784, 128),
    Tanh(),
    Dense(128, 64),
    Tanh(),
    Dense(64, 10),
    Softmax()
]
```

## Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. The following activation functions are implemented:

- **Tanh**: The hyperbolic tangent function, which outputs values between -1 and 1, and its derivative used during backpropagation.
- **Sigmoid**: The sigmoid function, which outputs values between 0 and 1, and its derivative (not used in this example but implemented).
- **ReLU**: The Rectified Linear Unit function, which outputs the input directly if positive and zero otherwise (not used in this example but implemented).
- **Softmax**: Converts the network output to a probability distribution by applying the exponential function and normalizing.

### Code Example

```python
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
```

## Loss Function

The Mean Squared Error (MSE) loss function measures the average squared difference between predicted and actual values. It is defined as:

$L(y_{\text{true}}, y_{\text{pred}}) = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true},i} - y_{\text{pred},i})^2$

where \( n \) is the number of samples. Its derivative with respect to the predictions is:

$L' = \frac{2 (y_{\text{pred}} - y_{\text{true}})}{n}$

### Code Example

```python
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
```

## Training

The network is trained using stochastic gradient descent, following these steps:

1. **Forward Pass**: Calculate the output of each layer by applying the weights, biases, and activation functions.
2. **Loss Calculation**: Compute the loss between the predicted output and the true labels.
3. **Backward Pass**: Calculate the gradients of the loss with respect to the weights and biases using backpropagation. Update the weights and biases using these gradients to minimize the loss.

### Code Example

```python
def train(network, loss, loss_prime, x_train, y_train, count=100, learning_rate=0.1):
    for e in range(count):
        error = 0
        for x, y in zip(x_train, y_train):
            output = x
            for layer in network:
                output = layer.forward_propagation(output) if isinstance(layer, Dense) else layer.forward(output)
            error += loss(y, output)
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward_propagation(grad, learning_rate) if isinstance(layer, Dense) else layer.backward(grad, learning_rate)
        print(f"Epoch {e + 1}/{count}, error={error}")
```

## Testing and Results

After training, the network's performance is evaluated on test samples. For each test sample:

1. **Prediction**: Obtain the network's predicted output.
2. **Visualization**: Display the test image and its predicted label.

### Code Example

```python
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('Actual Label: {}'.format(np.argmax(y)))
    plt.show()
    print('Predicted Label:', np.argmax(output))
```

## Usage

1. Install the required packages:
    ```bash
    pip install keras tensorflow
    ```
2. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo.git
    ```
3. Run the code:
    ```bash
    python your_script.py
    ```
