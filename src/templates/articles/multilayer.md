# **Building Neural Networks from Scratch: From Perceptron to XOR**

*This article builds upon the concepts introduced in [The Perceptron](/perceptron.html), extending single-layer learning to multi-layer neural networks.*

The perceptron we explored previously can solve linearly separable problems - cases where we can draw a straight line to separate two categories. But what happens when the data isn't linearly separable? This is where multi-layer neural networks shine, and the classic example that demonstrates their power is the XOR problem.

## **The XOR Problem: Why Single Layers Aren't Enough**

The XOR (exclusive OR) function represents a fundamental limitation of single-layer perceptrons. XOR outputs 1 when exactly one input is 1, but outputs 0 when both inputs are the same:

| A | B | XOR |
|:-:|:-:|:---:|
| 0 | 0 |  0  |
| 0 | 1 |  1  |
| 1 | 0 |  1  |
| 1 | 1 |  0  |

If you plot these points on a 2D plane, you'll discover something important: **no single straight line can separate the 1s from the 0s**. This non-linear separation requires a more sophisticated approach - multiple layers working together.

---

## **From Single Neurons to Neural Networks**

### The Multi-Layer Approach

A multi-layer neural network solves the XOR problem by combining multiple simple decision boundaries. Think of it this way:

1. **First layer**: Creates multiple linear decision boundaries
2. **Second layer**: Combines these boundaries to form complex, non-linear regions
3. **Output layer**: Makes the final classification

For XOR, we need at least one hidden layer with two neurons to create the necessary decision boundaries.

### Mathematical Foundation: Matrix Operations

While a single perceptron computes:
$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

A neural network layer processes multiple neurons simultaneously using matrix operations:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

Where:
- $\mathbf{a}^{[l]}$ is the output of layer $l$
- $\mathbf{W}^{[l]}$ is the weight matrix connecting layers
- $\mathbf{b}^{[l]}$ is the bias vector
- $f^{[l]}$ is the activation function

This matrix formulation allows us to compute all neurons in a layer simultaneously, making the implementation both efficient and elegant.

---

## **Designing Our Neural Network Framework**

Before diving into implementation details, let's design how we want to use our neural network library:

```python
# filename: main.py
from network import Network, LayerSpecNBF
import numpy as np
from sympy import Symbol, exp, tanh

def af_sigmoid(x: Symbol):
    return 1 / (1 + exp(-x))

def af_tanh(x: Symbol):
    return tanh(x)

# Define network architecture
nn = Network(
    inputs=2,
    layers=[
        LayerSpecNBF(2, True, af_tanh),    # Hidden layer: 2 neurons with bias
        LayerSpecNBF(2, True, af_tanh),    # Hidden layer: 2 neurons with bias  
        LayerSpecNBF(1, False, af_sigmoid), # Output layer: 1 neuron, no bias
    ]
)

# Training data for XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0],    [1],    [1],    [0]])

# Train the network
nn.train(X, y, epochs=10000, lr=0.015)
nn.export('xor.nn')
```

This design provides flexibility in specifying layer architectures while keeping the interface clean and intuitive.

---

## **Building the Network Infrastructure**

### Layer Specifications

Our framework supports multiple ways to specify layers, providing flexibility while maintaining type safety:

```python
# filename: network.py
import numpy as np
import numpy.typing as npt
from sympy import Symbol, diff, lambdify
from collections.abc import Callable, Sequence
from typing import cast, TypedDict, NamedTuple, Any
from pathlib import Path
import struct, marshal, types

LEARNING_RATE = 0.01

class LayerSpecNB(NamedTuple):
    length: int
    bias: bool

class LayerSpecNBF(NamedTuple):
    length: int
    bias: bool
    fn: Callable[[Symbol], Any] | tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]

class LayerSpecD(TypedDict):
    length: int
    bias: bool
    fn: Callable[[Symbol], Any] | tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]

LayerSpec = int | LayerSpecNB | LayerSpecNBF | LayerSpecD | tuple
```

### The Network Class

The Network class serves as the coordinator, managing layers and orchestrating the training process:

```python
class Network:
    def __init__(self, inputs: int, layers: list[LayerSpec] | list[Layer], fn=lambda f: f):
        """ The Network class. A coordinator of layers.
            
            Parameters:
            - inputs: The number of features of the neural network
            - layers: A list of LayerSpec or Layers
            - fn: The activation function to fallback to if none is specified in a layerspec
        """
        self.layers: list[Layer] = []
        self.inputs = inputs
        self.fn = fn

        if layers and all(map(lambda l: isinstance(l, Layer), layers)):
            self.layers = cast(list[Layer], layers)
        else:
            self.gen_layers(cast(list[LayerSpec], layers))

    def __repr__(self):
        return f"<Network:\n{'\n'.join(map(lambda s: f'\t{s!r}', self.layers))}\n>"

    def processSpec(self, spec: LayerSpec) -> LayerSpecNBF:
        if isinstance(spec, int):
            return LayerSpecNBF(spec, False, self.fn)
        if isinstance(spec, dict):
            return LayerSpecNBF(spec['length'], spec['bias'], spec['fn'])
        if isinstance(spec, tuple):
            if len(spec) == 2:
                return LayerSpecNBF(spec[0], spec[1], self.fn)
            if len(spec) == 3:
                return LayerSpecNBF(*cast(LayerSpecNBF, spec))
            raise TypeError(type(spec))
        if isinstance(spec, LayerSpecNB):
            return LayerSpecNBF(spec.length, spec.bias, self.fn)
        if isinstance(spec, LayerSpecNBF):
            return spec
        raise TypeError(type(spec))

    def train(self, X: np.ndarray, y: np.ndarray, epochs=100, lr: float=LEARNING_RATE):
        """ Train the network using the provided training data.
        
            Parameters:
            - X: Input training data (numpy array)
            - y: Target values (numpy array)
            - epochs: Number of training epochs
            - lr: Learning rate
        """
        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(y, np.ndarray): y = np.asarray(y)

        for epoch in range(epochs):
            for (inputs, target) in zip(X, y):
                self.back(inputs, target, lr=lr)
            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch} error {self.MSE(X, y)}")
                    
    def MSE(self, X: np.ndarray, y: np.ndarray):
        """Mean squared error loss function"""
        return np.sum((np.array([self(x).reshape(-1) for x in X]) - y) ** 2) / X.shape[0]

    def gen_layers(self, specs: list[LayerSpec]):
        prev = self.processSpec(self.inputs)
        for spec in specs:
            s = self.processSpec(spec)
            layer = Layer(s, prev)
            self.layers.append(layer)
            prev = s

    def export(self, fp: str | Path):
        with open(fp, 'wb') as f:
            f.write(struct.pack(">I", len(self.layers)))
            f.write(struct.pack(">I", self.inputs))
            for layer in self.layers:
                layer.export(f)

    @classmethod
    def load(cls, fp: str | Path):
        with open(fp, 'rb') as f:
            layers = []
            layer_count = struct.unpack(">I", f.read(4))[0]
            inputs = struct.unpack(">I", f.read(4))[0]
            for i in range(layer_count):
                layers.append(Layer.load(f))
            return cls(inputs, layers)
```

---

## **Forward Propagation: From Input to Output**

Forward propagation is the process of passing input data through the network to generate predictions. In our Network class, this is handled by the `__call__` method:

```python
class Network:
    def __call__(self, inputs: np.ndarray | Sequence[float], save=False):
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        # Force the input to be a column matrix for safety
        x = inputs.reshape(-1, 1)
        for layer in self.layers:
            x = layer(x)
        return x
```

This simple implementation demonstrates the power of our design: each layer transforms the input and passes it to the next layer, creating a computational pipeline.

---

## **The Layer Implementation**

### Automatic Differentiation with SymPy

One of the most elegant features of our implementation is automatic differentiation using SymPy. This eliminates the need to manually derive and implement derivatives:

```python
def prepfn(f: Callable[[Symbol], Any]):
    x = Symbol('x')
    f_sym = f(x)
    return (
        f_sym,                          # Symbolic function
        diff(f_sym, x),                 # Symbolic derivative
        lambdify(x, f_sym, 'math'),     # Python math function
        lambdify(x, diff(f_sym, x), 'math'), # Python math derivative
        lambdify(x, f_sym, 'numpy'),    # NumPy-compatible function
        lambdify(x, diff(f_sym, x), 'numpy'), # NumPy-compatible derivative
    )
```

This function takes a symbolic activation function (like `lambda x: 1/(1 + exp(-x))` for sigmoid) and automatically generates both the function and its derivative in multiple computational backends.

### The Layer Class

```python
class Layer:
    def __init__(self, spec: LayerSpecNBF, prev: LayerSpecNBF, *, weights: npt.NDArray[np.float64] | None=None, biases: npt.NDArray[np.float64] | None=None):
        self.length = spec.length
        self.prev = prev.length
        self.spec = spec
        self.elementwise = True

        if isinstance(spec.fn, tuple):
            self.f = self.cf = self.vf = spec.fn[0]
            self.df = self.cdf = self.dvf = spec.fn[1]
            self.elementwise = False
        else:
            self.f, self.df, self.cf, self.cdf, self.vf, self.dvf = prepfn(spec.fn)
            self.elementwise = True

        if weights is not None:
            self.weights: npt.NDArray[np.float64] = weights
        else:
            self.weights: npt.NDArray[np.float64] = np.random.randn(
                self.length, self.prev + int(self.spec.bias)) * np.sqrt(2 / (self.prev + int(self.spec.bias)))

        if biases is not None:
            self.biases: npt.NDArray[np.float64] = biases
        else:
            self.biases: npt.NDArray[np.float64] = np.zeros((self.length, 1))

        self.last_input = None
        self.z = None
        self.a = None

    def __call__(self, inputs: np.ndarray):
        if self.spec.bias:
            inputs = np.vstack([inputs, 1])
        self.last_input = inputs
        self.z = self.weights @ inputs + self.biases
        self.a = self.vf(self.z)
        return self.a

    def __repr__(self):
        return f"<Layer {self.weights.shape} {self.f}>"

    def back(self, grad, lr=LEARNING_RATE, momentum=0.9):
        if self.last_input is None or self.a is None or self.z is None:
            raise RuntimeError("The layer must be ran before training")
        if self.elementwise:
            dz = grad * self.dvf(self.z)
        else:
            dz = grad
        dw = dz @ self.last_input.T
        db = dz
        self.weights = self.weights - lr * dw
        self.biases = self.biases - lr * db

        grad_input = self.weights.T @ dz
        if self.spec.bias:
            grad_input = grad_input[:-1]
        return np.clip(grad_input, -1.0, 1.0)

    def export(self, fp: BinaryIO):
        fp.write(struct.pack(">I", self.spec.length))
        fp.write(struct.pack(">I", self.spec.bias))
        fp.write(struct.pack(">I", int(self.elementwise)))
        if self.elementwise:
            marshal.dump(self.spec.fn.__code__, fp) # pyright: ignore
        else:
            marshal.dump(self.spec.fn[0].__code__, fp) # pyright: ignore
            marshal.dump(self.spec.fn[1].__code__, fp) # pyright: ignore
        fp.write(struct.pack(">I", self.prev))

        np.save(fp, self.weights)
        np.save(fp, self.biases)

    @classmethod
    def load(cls, fp: BinaryIO):
        length = struct.unpack(">I", fp.read(4))[0]
        bias = struct.unpack(">I", fp.read(4))[0]
        elementwise = struct.unpack(">I", fp.read(4))[0]
        if elementwise:
            fn = types.FunctionType(marshal.load(fp), globals())
        else:
            f = types.FunctionType(marshal.load(fp), globals())
            df = types.FunctionType(marshal.load(fp), globals())
            fn = (f, df)
        prev = struct.unpack(">I", fp.read(4))[0]

        weights = np.load(fp)
        biases = np.load(fp)
        return cls(
            LayerSpecNBF(length, bias, fn),
            LayerSpecNBF(prev, False, lambda f:f),
            weights=weights,
            biases=biases,
        )
```

### Key Implementation Features

**Weight Initialization**: The layer uses He initialization (`np.sqrt(2 / fan_in)`) which works well with ReLU-like activation functions and helps prevent vanishing/exploding gradients.

**Forward Pass**: The `__call__` method implements the mathematical formula we discussed:
1. Adds bias term if specified
2. Computes pre-activation values: `z = W @ x + b`
3. Applies activation function: `a = f(z)`
4. Stores intermediate values for backpropagation

**Bias Handling**: Instead of maintaining separate bias vectors, the implementation appends a `1` to the input and includes bias weights in the main weight matrix. This simplifies the matrix operations.

---

## **Backpropagation: Learning from Mistakes**

Backpropagation is how neural networks learn - by propagating errors backward through the network and adjusting weights to minimize those errors.

### The Mathematics

The backpropagation algorithm uses the chain rule to compute gradients:

1. **Output gradient**: $\frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot f'^{[l]}(\mathbf{z}^{[l]})$

2. **Weight gradient**: $\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot (\mathbf{a}^{[l-1]})^T$

3. **Bias gradient**: $\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$

4. **Input gradient**: $\frac{\partial L}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{[l]}}$

### Layer-Level Backpropagation

The `back` method in our Layer class implements these equations:

```python
def back(self, grad, lr=LEARNING_RATE, momentum=0.9):
    if self.last_input is None or self.a is None or self.z is None:
        raise RuntimeError("The layer must be ran before training")
    
    # Compute gradient w.r.t. pre-activation values
    if self.elementwise:
        dz = grad * self.dvf(self.z)  # Chain rule with activation derivative
    else:
        dz = grad
    
    # Compute gradients w.r.t. weights and biases
    dw = dz @ self.last_input.T
    db = dz
    
    # Update parameters
    self.weights = self.weights - lr * dw
    self.biases = self.biases - lr * db

    # Compute gradient for previous layer
    grad_input = self.weights.T @ dz
    if self.spec.bias:
        grad_input = grad_input[:-1]  # Remove bias gradient
    
    return np.clip(grad_input, -1.0, 1.0)  # Gradient clipping for stability
```

### Network-Level Backpropagation

The Network class coordinates backpropagation across all layers:

```python
def back(self, inputs: np.ndarray | Sequence[float], target: np.ndarray | Sequence[float], lr=LEARNING_RATE):
    # Forward pass to compute predictions
    self(inputs)
    
    # Compute initial gradient (MSE loss derivative)
    if not isinstance(target, np.ndarray):
        target = np.asarray(target)
    grad = self.layers[-1].a - target.reshape(-1, 1)
    
    # Propagate gradients backward through all layers
    for layer in reversed(self.layers):
        grad = layer.back(grad, lr=lr)
```

This elegant implementation demonstrates how the chain rule propagates through the network:
1. Start with the error at the output
2. Each layer computes its gradients and updates its parameters
3. Each layer passes the gradient to the previous layer
4. The process continues until we reach the input

---

## **Solving XOR: Putting It All Together**

Now let's use our neural network to solve the XOR problem:

```python
# filename: xor_example.py
from network import Network, LayerSpecNBF
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, exp, tanh

def af_sigmoid(x: Symbol):
    return 1 / (1 + exp(-x))

def af_tanh(x: Symbol):
    return tanh(x)

def plot_decision_boundary(model, X, y):
    # Define the grid size and create the mesh grid
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = model(np.array([xx[i, j], yy[i, j]]))[0][0]
    
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.6)
    plt.colorbar()
    
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap=plt.cm.RdBu, edgecolors='k')
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title('XOR Decision Boundary')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    return plt

# XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0],    [1],    [1],    [0]])

# Create and train the network
nn = Network(
    inputs=2,
    layers=[
        LayerSpecNBF(2, True, af_tanh),    # Hidden layer with 2 neurons
        LayerSpecNBF(2, True, af_tanh),    # Another hidden layer
        LayerSpecNBF(1, False, af_sigmoid), # Output layer
    ],
)

nn.train(X, y, epochs=20000, lr=0.1)

# Test the network
print("\nTesting XOR Neural Network:")
print("--------------------------")
for inputs, target in zip(X, y):
    output = nn(inputs)
    print(f"Input: {inputs}, Target: {target[0]}, Predicted: {output[0][0]:.4f}")

# Visualize the decision boundary
plt.subplot(1, 2, 2)
plot_decision_boundary(nn, X, y)
plt.tight_layout()
plt.savefig('xor_training_results.png')
plt.show()

# Save the trained model
nn.export('xor.nn')
print("\nModel saved as 'xor.nn'")
```

### Understanding the Results

When you run this code, you should see the network successfully learn to classify XOR inputs. The decision boundary visualization will show how the network creates non-linear regions to separate the different classes - something impossible with a single-layer perceptron.

The key insight is that the hidden layers learn to create intermediate representations that make the final classification possible. Each hidden neuron learns a different linear boundary, and the output layer combines these to create the complex XOR decision boundary.

---

## **Key Insights and Next Steps**

### What We've Accomplished

1. **Built a complete neural network from scratch** using only NumPy and SymPy
2. **Implemented automatic differentiation** eliminating manual derivative calculation
3. **Demonstrated multi-layer learning** solving the classic XOR problem
4. **Created a flexible framework** supporting different architectures and activation functions

### The Power of Multi-Layer Networks

Our XOR solution demonstrates several crucial concepts:

- **Non-linear decision boundaries**: Multiple layers can solve problems that single layers cannot
- **Feature learning**: Hidden layers automatically learn useful intermediate representations
- **Gradient-based optimization**: Backpropagation efficiently trains complex networks
- **Composability**: Simple components (neurons) combine to solve complex problems

### Looking Forward

This foundation opens the door to understanding modern deep learning:

- **Deeper networks**: More layers can learn more complex patterns
- **Different architectures**: CNNs, RNNs, Transformers all build on these principles
- **Advanced optimizers**: Adam, RMSprop improve on basic gradient descent
- **Regularization**: Dropout, batch normalization improve training

The neural network you've built here contains the essential DNA of every modern deep learning system. While production frameworks like PyTorch and TensorFlow add many optimizations and conveniences, the core principles remain the same: forward propagation, backpropagation, and gradient descent working together to learn from data.

Understanding these fundamentals gives you the insight needed to debug training issues, design new architectures, and push the boundaries of what's possible with neural networks.


---

# Bonus

As a "showcase", here is a small javascript neural network implementation and it's decision boundary.
```javascript
class NeuralNetwork {
	constructor() {
		this.w1 = Array(2).fill().map(() => Array(2).fill().map(() => Math.random() - 0.5));
		this.w2 = Array(2).fill().map(() => Math.random() - 0.5);
		this.b1 = Array(2).fill(0);
		this.b2 = 0;
		this.learningRate = 3;
	}

	sigmoid(x) {
		return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
	}

	sigmoidDerivative(x) {
		return x * (1 - x);
	}

	forward(input) {
		this.z1 = Array(2);
		this.a1 = Array(2);
		for (let i = 0; i < 2; i++) {
			this.z1[i] = this.w1[i][0] * input[0] + this.w1[i][1] * input[1] + this.b1[i];
			this.a1[i] = this.sigmoid(this.z1[i]);
		}
		this.z2 = this.w2[0] * this.a1[0] + this.w2[1] * this.a1[1] + this.b2;
		this.a2 = this.sigmoid(this.z2);
		return this.a2;
	}

	train(input, target) {
		const output = this.forward(input);
		const error = target - output;
		const delta2 = error * this.sigmoidDerivative(output);
		const delta1 = Array(2);
		for (let i = 0; i < 2; i++)
			delta1[i] = delta2 * this.w2[i] * this.sigmoidDerivative(this.a1[i]);
		for (let i = 0; i < 2; i++)
			this.w2[i] += this.learningRate * delta2 * this.a1[i];
		this.b2 += this.learningRate * delta2;
		for (let i = 0; i < 2; i++) {
			for (let j = 0; j < 2; j++)
				this.w1[i][j] += this.learningRate * delta1[i] * input[j];
			this.b1[i] += this.learningRate * delta1[i];
		}
	}

	predict(input) {
		return this.forward(input);
	}
}
```
