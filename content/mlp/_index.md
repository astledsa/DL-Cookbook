---
title: "Multi-layer Perceptron"
date: 2023-04-20
draft: false
references:
  - title: "Grokking Deep learning"
    url: "https://edu.anarcho-copy.org/Algorithm/grokking-deep-learning.pdf"
  - title: "What is a neural network?"
    url: "https://youtu.be/aircAruvnKk?si=ZraEDT94knQqAs3M"
  - title: "Approximation by Superpositions of a Sigmoidal Function"
    url: "https://web.njit.edu/~usman/courses/cs677/10.1.1.441.7873.pdf"
---

The central building block of any deep learning architecture. One of the main feature of MLPs, that essentially separeted it from classical Machine Learning was the use of hidden layers, which worked well with the gradient based optimization called backpropagation. 

<br>

## Equation

![MLP Diagram](/images/mlp2.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>X</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W<sub>1</sub></strong></td>
    <td style="vertical-align: middle;">: First weight matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b<sub>1</sub></strong></td>
    <td style="vertical-align: middle;">: First bias vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜎</strong></td>
    <td style="vertical-align: middle;">: Activation Function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W<sub>2</sub></strong></td>
    <td style="vertical-align: middle;">: Second weight matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b<sub>2</sub></strong></td>
    <td style="vertical-align: middle;">: Second bias vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output vector</td>
  </tr>
</table>

<br>

## Explanation

- One of the main features of this particular structure, alongside backpropagation that allowed for its widespread use was the activation function, which allowed it to discriminate continuous function spaces into different subsets through a non-linear decision boundary, and have its own "internal representation" of the data (through the hidden layer). This particular feature contributed significantly to its distinction from classical ML algorithms, gradient-based learning and otherwise, that led to its theoretical power, as shown by the universal approximation theorems [3] demonstrated by Cybenko, Hornik, and others.

- While many resources would claim that the original inspiration was our brain, it is only partially correct. The mathematical background came from the method of least squares, aka linear regression, which shares some mathematical principles with single-layer perceptrons, and an analogy was then made to the brain and its neurons. Modern deep learning practitioners rarely make that connection in recent literature, as it can be misleading and often unnecessary. Rather than a biological framework, a computational one would be more beneficial for research and understanding (as is implicitly suggested in the [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) by Rich Sutton).

<br>

## Code

```python
 import jax
 from jax import random
 import jax.numpy as jnp

 # Initialise a random key
 key = random.PRNGKey(100)
 key1, key2, key3, key4, key5 = random.split(key, num=5)

 # The dimension of our hidden layer
 hidden_size = 100

 # A random vector as an input
 x  = random.normal(key1, (1, 100))

 # Initialise the weights or parameters the required dimensions
 w1 = random.normal(key2, (100, hidden_size))
 w2 = random.normal(key3, (hidden_size, 100))
 b1 = random.normal(key4, (1, 100))
 b2 = random.normal(key5, (1, 100))

 # h = 𝜎(w1@x + b1)
 def first_layer (w1, b1, x) :
   return jax.nn.relu(jnp.dot(x, w1) + b1)

 # y = w2@h + b2
 def hidden_layer (w2, b2, h) :
   return jnp.dot(h, w2) + b2 

 y = hidden_layer(w2, b2, first_layer(w1, b1, x))
```