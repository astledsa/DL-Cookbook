---
title: "ResNet"
date: 2025-03-30
draft: false
references:
  - title: "Deep Residual Learning for Image Recognition"
    url: "https://arxiv.org/abs/1512.03385s"
---

One of the most important innovations in the general training regime of deep neural networks was the introduction of "skip" connections, which eventually enabled the training of larger and larger models by solving one of the more persistent problems (vanishing gradients) and paved the way for further advancements. Residual networks.

<br>

## Equation

\begin{align}
\sigma(XW_{1} + b_1) + X = y
\end{align}

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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output vector</td>
  </tr>
</table>

<br>

## Explanation

- The idea is extremely simple, yet extremely effective. Vanishing gradient problem ? Just add back the input after a few layers ! While in retrospect this seems rather naive, it works well empirically, and the theoretical foundations simply add a cherry on top. Instead of learning a continuos functions (which the network can do, in theory), it is instead forced to learn the residual instead. The main advantage is in making the network learn what is close to the identity mapping function between our input and the output of the network layer.

- Once we take the *gradients* of the relevant equations, we come across the derivative of the input with itself, which equals to one. This case, where no matter what the other output is, we always end up getting *at least* a one, effectively avoids the vanishing gradient problem, as the information (which is the gradients) about the identity function is being passed backward (through the one). This leads to us being able to train the models at large scales.

<br>

## Code

```python
import jax
import jax.numpy as jnp

def residual_connection(X, W1, b1):
  linear_transform = jnp.dot(X, W1) + b1
  activation = jax.nn.sigmoid(linear_transform)
  y = activation + X # It is that simple !
  return y

key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (128, 256)) 
W1 = jax.random.normal(key, (256, 128)) 
b1 = jax.random.normal(key, (128,))

y = residual_connection(X, W1, b1)
```