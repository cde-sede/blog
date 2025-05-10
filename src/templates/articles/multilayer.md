# Multilayer neural network

The goal of this article is to write a neural network class to act as an [xor gate](https://en.wikipedia.org/wiki/XOR_gate).

## The goal

Let's start by hallucinating how we'd like to use our library

```python
# filename: main.py
from network import Network, Layer, LayerSpecNBF
from activation import sigmoid, tanh
import numpy as np

nn = Network(
    inputs = 2,
    specs = [
        LayerSpecNBF(2, True, tanh),
        LayerSpecNBF(2, True, tanh),
        LayerSpecNBF(1, True, sigmoid),
    ],
)

X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1], ])
y = np.array([ [0],    [1],    [1],    [0],])

nn.train(X, y, epochs=1000, lr=0.1)
nn.export('xor.nn')
```

## The Network class

```python
# filename: network.py
import numpy as np
from sympy import Symbol
from collections.abc import Callable, Sequence
from typing import cast, TypedDict, NamedTuple, Any

class LayerSpecNB(NamedTuple):
	length: int
	bias: bool

class LayerSpecNBF(NamedTuple):
	length: int
	bias: bool
	fn: Callable[[Symbol], Any] | tuple[Callable[[np.ndarray], np.ndarray],Callable[[np.ndarray], np.ndarray]]

class LayerSpecD(TypedDict):
	length: int
	bias: bool
	fn: Callable[[Symbol], Any] | tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]

LayerSpec = int | LayerSpecNB | LayerSpecNBF | LayerSpecD | tuple

class Network:
	def __init__(self, inputs: int, layers: list[LayerSpec] | list[Layer], fn=lambda f: f):
		self.layers: list[Layer] = []
		self.inputs = inputs
		self.fn = fn

		if layers and isinstance(layers[0], Layer):
			self.layers = cast(list[Layer], layers)
		else:
			self.gen_layers(cast(list[LayerSpec], layers))

	def __repr__(self):
		return f"<Network:\n{"\n".join(map(lambda s: f"\t{s!r}", self.layers))}\n>"

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

The code currently features two key components.

### Layer Specification Types
The code defines several ways to specify neural network layers:
- Using a simple integer (just specifying the number of nodes)
- Named tuples (`LayerSpecNB` and `LayerSpecNBF`) for structured specifications, like bias and the activation function
- TypedDict (`LayerSpecD`) for dictionary-based specifications
- Tuples of varying length, assuming the same sturcture as Named tuples

This flexibility allows the user to define network architectures in whatever format is most convenient.

### The Container
The Network class serves as the container for layers and provides methods to:
1. Initialize a network with custom layer specifications
2. Generate layers based on specifications
3. Export the network to a file
4. Load a network from a file

### What's Missing

This code is missing two very important parts, the Layer implementation, and the forward and backward propagation.

Let's start with the forward propagation. Building on the perceptron, each individual neural's value is the activation function applied on the weighted sum of the previous layer. Letting the `Layer` class handle the computation, the responsiblity of the Network is to propagate the output of each layer to the next.


```python
# filename: network.py
class Network:
    ...
	def __call__(self, inputs: np.ndarray | Sequence[float], save=False):
		if not isinstance(inputs, np.ndarray):
			inputs = np.asarray(inputs)
        # Force the input to be a column matrix for safety,
        # but really if it is not the input should be sanitized
		x = inputs.reshape(-1, 1)
		for layer in self.layers:
			x = layer(x)
		return x
```


## The layer implementation


```python
# filename: network.py
def prepfn(f: Callable[[Symbol], Any]):
	x = Symbol('x')
	f_sym = f(x)
	return (
		f_sym,
		diff(f_sym, x),
		lambdify(x, f_sym, 'math'),
		lambdify(x, diff(f_sym, x), 'math'),
		lambdify(x, f_sym, 'numpy'),
		lambdify(x, diff(f_sym, x), 'numpy'),
	)

class Layer:
	def __init__(self, spec: LayerSpecNTF, prev: LayerSpecNTF, *, weights: npt.NDArray[np.float64] | None=None, biases: npt.NDArray[np.float64] | None=None):
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
			LayerSpecNTF(length, bias, fn),
			LayerSpecNTF(prev, False, lambda f:f),
			weights=weights,
			biases=biases,
		)
```


The `Layer` class represents a single neural network layer with weights, biases, and activation functions. What makes this implementation special is its use of symbolic computation via SymPy alongside NumPy for numerical operations.

### Function Preparation with Symbolic Differentiation

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

This function uses SymPy's symbolic differentiation capabilities to automatically generate the derivative of any activation function. The `lambdify` function then converts these symbolic expressions into callable functions compatible with both Python's math module and NumPy arrays.

### Layer Initialization

The `Layer` class constructor accepts a specification for the current layer, information about the previous layer, and optional pre-defined weights and biases:

```python
def __init__(self, spec: LayerSpecNTF, prev: LayerSpecNTF, *, 
             weights: npt.NDArray[np.float64] | None=None, 
             biases: npt.NDArray[np.float64] | None=None):
```

The constructor handles two types of activation functions:
1. Pre-compiled function/derivative pairs (when `spec.fn` is a tuple)
2. SymPy symbolic functions that get automatically differentiated (via `prepfn`)

When weights aren't provided, they're initialized using the He initialization method (`np.random.randn(...) * np.sqrt(2 / (self.prev + int(self.spec.bias)))`), which is optimized for layers with ReLU-like activation functions.

### Serialization and Deserialization

The implementation includes methods to save and load layers:

```python
def export(self, fp: BinaryIO):
    # Save layer configuration and parameters to a binary file
    
@classmethod
def load(cls, fp: BinaryIO):
    # Load layer configuration and parameters from a binary file
```

What's notable here is the use of Python's `marshal` module to serialize and deserialize activation functions - a sophisticated approach that allows the entire network, including its custom functions, to be saved and restored.


### Forward Pass

The forward pass is implemented in the `__call__` method, allowing layers to be used as functions:

```python
def __call__(self, inputs: np.ndarray):
    if self.spec.bias:
        inputs = np.vstack([inputs, 1])
    self.last_input = inputs
    self.z = self.weights @ inputs + self.biases
    self.a = self.vf(self.z)
    return self.a
```

This method:
1. Adds a bias term to the input if specified by the LayerSpec
2. Stores the input for use in backpropagation
3. Computes the pre-activation values (`z`)
4. Applies the activation function
5. Returns the activated output




In a previous [article](/perceptron.html), we explored the simple perceptron model, which provides the foundation for neural networks. Now, we'll extend this understanding to explain how forward propagation works in multi-layer networks using matrix operations, as implemented in our neural network framework.

Recall that a single perceptron computes its output as follows:

{{previous}}

Where:
- {{inputs}} are the inputs
- {{weights}} are the weights
- {{bias}} is the bias
- {{fn}} is the activation function

## Extending to Layers with Matrix Operations

The neural network layer implementation we examined earlier extends this concept to handle multiple neurons organized in layers, using matrix operations for efficient computation.

### Mathematical Formulation

For a layer with $m$ neurons receiving input from $n$ neurons in the previous layer, we can express the forward propagation as:

$$\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$$

Where:

 $$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

In these equations:
- ~$\mathbf{a}^{[l]}$ is the activation (output) vector of layer $l$
- ~$\mathbf{z}^{[l]}$ is the pre-activation vector of layer $l$
- ~$\mathbf{W}^{[l]}$ is the weight matrix of layer $l$
- ~$\mathbf{b}^{[l]}$ is the bias vector of layer $l$
- ~$f^{[l]}$ is the activation function of layer $l$
- ~$\mathbf{a}^{[l-1]}$ is the activation (output) vector of the previous layer

The dimensions of these elements are:
- ~$\mathbf{W}^{[l]}$ is an $m \times n$ matrix
- ~$\mathbf{a}^{[l-1]}$ is an $n \times 1$ vector
- ~$\mathbf{b}^{[l]}$ is an $m \times 1$ vector
- ~$\mathbf{z}^{[l]}$ is an $m \times 1$ vector
- ~$\mathbf{a}^{[l]}$ is an $m \times 1$ vector

### Implementation in Code

This is precisely what we see in the `__call__` method of the `Layer` class:

```python
def __call__(self, inputs: np.ndarray):
    if self.spec.bias:
        inputs = np.vstack([inputs, 1])
    self.last_input = inputs
    self.z = self.weights @ inputs + self.biases
    self.a = self.vf(self.z)
    return self.a
```

Let's break down how this code implements the mathematical formulation:

1. **Bias Handling**: Instead of adding a separate bias term, the code appends a 1 to the input vector and incorporates the bias into the weight matrix:
   ```python
   if self.spec.bias:
       inputs = np.vstack([inputs, 1])
   ```
   This transforms our equation to:
   {{biased}}

   Where:
   {{biased2}}

2. **Matrix Multiplication**: The actual computation uses matrix multiplication (the `@` operator in Python):
   ```python
   self.z = self.weights @ inputs + self.biases
   ```
   This calculates $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, the pre-activation values.

3. **Activation Function**: The activation function is applied element-wise to the pre-activation values:
   ```python
   self.a = self.vf(self.z)
   ```
   This computes $\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$, the final output of the layer.

### Why Matrix Operations?

The matrix-based approach offers several advantages over element-wise computation:

1. **Computational Efficiency**: Matrix operations are highly optimized in libraries like NumPy, leveraging hardware acceleration.
2. **Code Clarity**: The matrix formulation makes the code more concise and readable.
3. **Batch Processing**: Matrices allow us to process multiple samples simultaneously.
4. **Vectorization**: Eliminates explicit loops, reducing computational overhead.

### The Power of Vectorization

In practice, we typically process multiple samples simultaneously. If we have a batch of $k$ samples, we can extend our formulation:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \cdot \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = f^{[l]}(\mathbf{Z}^{[l]})$$

Where:
- ~$\mathbf{A}^{[l-1]}$ is an $n \times k$ matrix containing activations for all samples
- ~$\mathbf{Z}^{[l]}$ is an $m \times k$ matrix of pre-activation values
- ~$\mathbf{A}^{[l]}$ is an $m \times k$ matrix of activation values


Now, the only thing left to do is the training part of the network: backpropagation.

## The Backpropagation Algorithm: Mathematical Foundation

Backpropagation is based on the chain rule from calculus, allowing us to compute how changes in the network's parameters affect the overall error. Let's develop the math step by step.

### The Chain Rule in Neural Networks

For a network with a loss function $L$, we want to compute how changes in weights and biases affect this loss. Given the layered structure of neural networks, we use the chain rule to propagate gradients backward.

For a layer $l$, we need to compute:

1. ~$\frac{\partial L}{\partial \mathbf{W}^{[l]}}$ - How changes in weights affect the loss
2. ~$\frac{\partial L}{\partial \mathbf{b}^{[l]}}$ - How changes in biases affect the loss
3. ~$\frac{\partial L}{\partial \mathbf{a}^{[l-1]}}$ - How changes in the previous layer's output affect the loss

### Key Backpropagation Equations

Starting with a gradient $\frac{\partial L}{\partial \mathbf{a}^{[l]}}$ received from the next layer (or directly from the loss function for the output layer), we compute:

1. ~$\frac{\partial L}{\partial \mathbf{z}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \odot f'^{[l]}(\mathbf{z}^{[l]})$ for elementwise activation functions  
   Where $\odot$ represents element-wise multiplication

2. ~$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot (\mathbf{a}^{[l-1]})^T$  
   This gives us the gradient for the weights

3. ~$\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$  
   The gradient for the biases

4. ~$\frac{\partial L}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{[l]}}$  
   This gradient is passed to the previous layer

### Parameter Update Rule

Once we have the gradients, we update the parameters using gradient descent:

$$\mathbf{W}^{[l]} = \mathbf{W}^{[l]} - \alpha \cdot \frac{\partial L}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} = \mathbf{b}^{[l]} - \alpha \cdot \frac{\partial L}{\partial \mathbf{b}^{[l]}}$$

Where $\alpha$ is the learning rate.

## Implementation in Code

Now, let's examine how these mathematical principles are implemented in our neural network framework:

```python
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
```

Let's break down how this implementation corresponds to the mathematical equations:

### 1. Computing the Pre-Activation Gradient

```python
if self.elementwise:
    dz = grad * self.dvf(self.z)
else:
    dz = grad
```

This computes $\frac{\partial L}{\partial \mathbf{z}^{[l]}}$. For elementwise activation functions, it multiplies the incoming gradient (`grad`, which is $\frac{\partial L}{\partial \mathbf{a}^{[l]}}$) by the derivative of the activation function evaluated at the pre-activation values.

For non-elementwise functions, it assumes the incoming gradient already accounts for the activation function's effect.

### 2. Computing the Weight Gradient

```python
dw = dz @ self.last_input.T
```

This computes $\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot (\mathbf{a}^{[l-1]})^T$. The matrix multiplication (`@` operator) between the pre-activation gradient and the transpose of the layer's input gives us the gradient with respect to the weights.

### 3. Computing the Bias Gradient

```python
db = dz
```

This sets $\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$, as the bias gradient is simply the pre-activation gradient.

### 4. Updating Parameters

```python
self.weights = self.weights - lr * dw
self.biases = self.biases - lr * db
```

This implements the gradient descent update rule:
- ~$\mathbf{W}^{[l]} = \mathbf{W}^{[l]} - \alpha \cdot \frac{\partial L}{\partial \mathbf{W}^{[l]}}$
- ~$\mathbf{b}^{[l]} = \mathbf{b}^{[l]} - \alpha \cdot \frac{\partial L}{\partial \mathbf{b}^{[l]}}$

Where `lr` corresponds to the learning rate $\alpha$.

### 5. Computing the Gradient for the Previous Layer

```python
grad_input = self.weights.T @ dz
if self.spec.bias:
    grad_input = grad_input[:-1]
```

This computes $\frac{\partial L}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \cdot \frac{\partial L}{\partial \mathbf{z}^{[l]}}$, the gradient to be passed to the previous layer.

If the layer uses a bias term, we need to remove the corresponding gradient element since it was added artificially during forward propagation.

### 6. Gradient Clipping

```python
return np.clip(grad_input, -1.0, 1.0)
```

This implements gradient clipping, a technique to prevent exploding gradients by constraining them to a specific range. This improves training stability, especially for deep networks.

## Network backpropagation

After examining the backpropagation mechanics at the layer level, we now explore how the entire network coordinates this learning process. The network-level backpropagation orchestrates gradient flow through multiple layers, ensuring that each component of the network learns its optimal parameters.

### The Network's Backpropagation Method

```python
def back(self, inputs: np.ndarray | Sequence[float], target: np.ndarray | Sequence[float], lr=LEARNING_RATE):
    self(inputs)
    if not isinstance(target, np.ndarray):
        target = np.asarray(target)
    grad = self.layers[-1].a - target.reshape(-1, 1)
    for layer in reversed(self.layers):
        grad = layer.back(grad, lr=lr)
```

Though remarkably concise, this method encapsulates the essence of neural network training. Let's dissect its components and understand the mathematical principles at work.

### Mathematical Analysis of Network Backpropagation

#### 1. Forward Pass Initialization

```python
self(inputs)
```

Before backpropagation can begin, the network must compute the forward pass. This single line invokes the `__call__` method of the Network class, which propagates the input through all layers. This ensures that each layer has properly calculated and stored its intermediate values (`last_input`, `z`, and `a`), which are essential for backpropagation.

#### 2. Error Calculation with Mean Squared Error (MSE)

```python
grad = self.layers[-1].a - target.reshape(-1, 1)
```

This line computes the initial gradient based on the Mean Squared Error (MSE) loss function. Mathematically, the MSE loss is:

{{mseloss}}

Where:
- ~$y_i$ is the predicted output (here, `self.layers[-1].a`)
- ~$\hat{y}_i$ is the target output (here, `target`)
- ~$m$ is the number of output units

The gradient of this loss with respect to the output activations is:

{{grad}}

The code computes a simplified version of this gradient as $(y_i - \hat{y}_i)$, omitting the constants which would only scale the learning rate. The `.reshape(-1, 1)` ensures proper dimensionality for column vector operations.

#### 3. Backward Gradient Propagation

```python
for layer in reversed(self.layers):
    grad = layer.back(grad, lr=lr)
```

This loop iterates through the layers in reverse order, starting from the output layer and moving toward the input layer. This reverse traversal is central to the backpropagation algorithm, as it allows gradients to flow backward through the network.

For each layer:
1. The current gradient (`grad`) is passed to the layer's `back` method
2. The layer updates its weights and biases based on this gradient
3. The layer computes and returns the gradient for the previous layer
4. This returned gradient becomes the input for the next iteration of the loop

### The Chain Rule in Action

The entire backpropagation algorithm is a practical application of the chain rule from calculus. For a deep network with $L$ layers, we're calculating:

$$\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial a^{[L-1]}} \cdot \ldots \cdot \frac{\partial a^{[2]}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} \cdot \frac{\partial a^{[1]}}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial W^{[1]}}$$

The beauty of backpropagation is that it calculates these gradients efficiently by working backward. Each layer in the loop:
1. Receives $\frac{\partial L}{\partial a^{[l]}}$ (the gradient with respect to its output)
2. Calculates $\frac{\partial L}{\partial W^{[l]}}$ and $\frac{\partial L}{\partial b^{[l]}}$ to update its parameters
3. Calculates $\frac{\partial L}{\partial a^{[l-1]}}$ to pass to the previous layer
