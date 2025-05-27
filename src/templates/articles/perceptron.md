# The Perceptron

The Perceptron is an algorithm of supervised learning. Originally presented in the book Perceptrons: An Introduction to Computational Geometry by Marvin Minksy and Seymour Papert. It is a type of artificial neural network, whose goal is binary classification - Deciding in which category the output falls.
In essence, a Perceptron is exactly the same a linear regression with gradient descent.


## The Problem

Let's say we have to decide whether an arbitrary point **p** falls under or above a line. Assuming we know the equation of the line this is quite simple.
We simply use the line equation

$$ap_x + b > p_y$$

Where **a** and **b** are the slope and y-intercept.  
Now let's try without this information, instead we have a collection of points and whether or not they are above a line. We have two solutions here: Linear Regression or a Neural Network

### Linear Regression

Linear Regression is quite simple in concept.  
It works by minimizing the sum of **squared differences** between the predicted values and the actual values.  
In other words, the goal is to find the values **a** and **b** that minimizes the **error** between our guess and the real line.  

We have a set of data points (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ). 

For each point, the error is the difference between the actual y value and our prediction:
- Error for a point = actual y - predicted y
- Error for a point = yᵢ - (m×xᵢ + b)

#### Defining What "Best Fit" Means

We define the "best" line as the one that minimizes the sum of squared errors:

$$\frac{1}{n}\sum{(y_i - (ax_i+b))^2}$$

We square the errors because:
- It penalizes larger errors more heavily
- It treats positive and negative errors equally
- It makes the math more tractable by creating a smooth function to minimize

#### Finding the Minimum Error

To find the minimum of the error function, we use calculus. We take the partial derivatives of the error function with respect to m and b, and set them to zero:

$$\begin{align*} \partial(Error)/\partial{a} = 0 \\ \partial(Error)/\partial{b} = 0\end{align*}$$

This gives us two equations (called "normal equations"):

$$\begin{align*} \partial(Error)/\partial{a} = -2\sum{x_i (y_i - (ax_i + b))} = 0 \\ \partial(Error)/\partial{b} = -2\sum{y_i - (ax_i + b)} = 0\end{align*}$$

#### Solving the System of Equations

Simplifying these equations:

For the intercept b:
$$\begin{align*} \sum{(y_i - ax_i - b)} = 0 \\ \sum{y_i - a\sum{x_i - nb}} = 0 \\ b = \frac{\sum{y_i - a\sum{x_i}}}{n} \\ b = \bar{y} - a\bar{x} \end{align*}$$

(where $\bar{y}$ and $\bar{x}$ are the means and n the number of points)

For the slope m:
$$\begin{align*} \sum(x_i(y_i - ax_i - b)) = 0 \\ \sum(x_{i}y_{i}) - a\sum{x_{i}^2} - b\sum{x+i} = 0 \end{align*}$$

Substituting the expression for b:
$$\begin{equation} \begin{aligned} \sum (x_i y_i) - a \sum (x_i^2) - (\bar{y} - a \bar{x}) \sum x_i &= 0 \\ \sum (x_i y_i) - a \sum (x_i^2) - \bar{y} \sum x_i + a \bar{x} \sum x_i &= 0 \\ \sum (x_i y_i) - \bar{y} \sum x_i &= a \sum (x_i^2) - a \bar{x} \sum x_i \\ \sum (x_i y_i) - \bar{y} \sum x_i &= a \left( \sum (x_i^2) - \bar{x} \sum x_i \right) \end{aligned} \end{equation}$$

This can be simplified to:
$$a = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

#### The Final Solution

With some algebraic manipulation, we get these cleaner formulas:
$$\begin{align*} a = \frac{n \sum (x_i y_i) - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2} \\ b = \bar{y} - a \bar{x} = \frac{\sum y_i - a \sum x_i}{n} \end{align*}$$

#### Why This Works

This approach works because:

1. **Mathematical Guarantee**: The calculus-based approach guarantees we find the **global minimum** of the squared error function.

2. **Unique Solution**: For linear regression, there is exactly one line that minimizes the sum of squared errors.

3. **Optimal Predictions**: The resulting line gives us predictions that are as close as possible to the actual data (in terms of squared error).

4. **Statistical Properties**: Under certain assumptions, these estimates have desirable statistical properties (like being unbiased).

Linear regression finds the mathematically optimal solution without needing to guess or iterate, unlike gradient descent which approaches the solution gradually. When the dataset is small to medium-sized, this direct solution is efficient and precise.

Here is an implementation of the linear regression:  


```python
# filename: LinearRegression.py
class LinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, x_values, y_values):
        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n
        
        sum_xy = 0
        sum_xx = 0
        for i in range(n):
            sum_xy += x_values[i] * y_values[i]
            sum_xx += x_values[i] * x_values[i]
        
        numerator = n * sum_xy - sum(x_values) * sum(y_values)
        denominator = n * sum_xx - sum(x_values) ** 2
        
        if denominator == 0:
            raise ValueError("Cannot fit a line with zero slope denominator")
            
        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x
        
        return self
    
    def predict(self, x_values):
        predictions = []
        for x in x_values:
            y_pred = self.slope * x + self.intercept
            predictions.append(y_pred)
            
        return predictions
```

#### Gradient descent

Here we are analytically calculating the minimum of the function. But it is also possible to approximate it by trial and error.  
Gradient descent is the idea of minimizing the error by going towards the local minima using the derivative of the function. 

I won't go over it in too much detail, as an exercise you could try to implement it yourself.  
Here is a working version: (click on the code block to reveal the code)

```python
# filename: LinearRegressionGradientDescent.py
# spoiler: true
class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def _mean_squared_error(self, X, y, weights, bias):
        return np.mean((pnp.dot(X, weights) + bias - y) ** 2)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            predictions = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### Neural Networks

While this approach works and allows us to get a continuous answer, there are still a few issues:
- The result can only be linear
- Heteroscedasticity
- Outliers have higher leverage due to the square

To some degree, some of these issues can be mitigated by changing the approach, adding dimensions, doing some statistical analysis on the dataset etc.
But these solutions have to be fine tuned to the dataset and the problem.

Furthermore on large datasets Regression can be very memory intensive.  
The solution is to compute the **local minimum** of the function and include some non linear functions into the computation.

At the core of training a neural network is a simple 3 steps idea

- Make a prediction
- Calculate by how much the prediction is off, and in which direction to go to reduce the error
- Change how the prediction

Repeat until satisfaction.  
Let's see how each step goes in the case of a Perceptron

#### 1. The prediction

A perceptron is composed of 3 parts
- **Inputs**: The values we use to make the prediction
- **Weights**: How much each input changes the prediction
- **Activation** Function: A function to introduce some non linearity (for example the **sigmoid** $f(x) = \frac{1}{1+e^x}$)

To make the prediction, we compute the weighted sum of inputs, then feed this value into the activation function

$$pred = f(\sum{x_i w_i})$$

#### 2. The error

Once we have a prediction, we compare it to the expected output using a **loss function**.
A common choice is **mean squared error (MSE)**:

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Or simply the distance between the prediction and the expected output $L = y - \hat{y}$

This gives us a measure of how far off our predictions are. The goal is to minimize this loss.

#### 3. The correction

To reduce the error, we adjust the weights.
We compute the **gradient** of the loss function with respect to each weight — how much a small change in a weight would change the loss.
This is done via **backpropagation**, and we use **gradient descent** to update the weights:

$$ w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w} $$

Where $\eta$ is the learning rate — how big each step is.

Repeat this loop (predict → compute error → update weights) until the model converges or reaches a stopping criterion.

```python
# spoiler: true
# filename: perceptron.py
from itertools import starmap
from operator import mul
import random

SIGN = lambda f: (f > 0) - (f < 0)
XMIN, XMAX = -1, 1
YMIN, YMAX = -1, 1
WIDTH = 400
HEIGHT = 400
NUM_POINTS = 2000
LEARNING_RATE = 0.001
POINTS = [(
	random.random() * 2 - 1,
	random.random() * 2 - 1,
	) for _ in range(NUM_POINTS)
]

LINE = lambda x: 0.3 * x + 0.4

DATASET = [
	[(*p, 1), SIGN(LINE(p[0]) - p[1])] for p in POINTS
]

class Perceptron:
	def __init__(self, num_inputs):
		self.weights = [random.random() for _ in range(num_inputs)]
		self.activation = SIGN

	def guess(self, values):
		return self.activation(sum(starmap(mul, zip(self.weights, values))))

	def train(self, inputs, target):
		guess = self.guess(inputs)

		error = target - guess
		self.weights = [w + error * i * LEARNING_RATE for w, i in zip(self.weights, inputs)]


perceptron = Perceptron(3)
for epoch in range(10):
    for x, y in DATASET:
        perceptron.train(x, y)

print(f"1st Point: {DATASET[0][0]} correct: {DATASET[0][1]} guess: {perceptron.guess(DATASET[0][0])}")
print(f"2nd Point: {DATASET[1][0]} correct: {DATASET[1][1]} guess: {perceptron.guess(DATASET[1][0])}")
print(f"3rd Point: {DATASET[2][0]} correct: {DATASET[2][1]} guess: {perceptron.guess(DATASET[2][0])}")
```


And using numpy

```python
# filename: perceptron_np.py
import numpy as np

class Perceptron:
	def __init__(self, num_inputs):
		self.weights = np.asarray([random.random() for _ in range(num_inputs)], dtype=np.float64)
		self.activation = np.sign

	def guess(self, values):
		return self.activation(np.sum(self.weights * np.asarary(values)))

	def train(self, inputs, target):
		guess = self.guess(inputs)

		error = target - guess
		self.weights += error * inputs * LEARNING_RATE
```
