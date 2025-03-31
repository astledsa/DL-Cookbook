---
title: "Gaussian Adaptive Transformers"
date: 2024-12-10
draft: false
references:
  - title: "Gaussian Adaptive Attention is all you need: Robust contexual representations across multiple modalitites"
    url: "https://arxiv.org/abs/2401.11143v3"  
---

As transformers became more and more popular, research into finding better and more efficient variants of the said architecture exploded, and various new modifications were made over the past few years. (Which makes writing an up-to-date book on deep learning that much troublesome!)

<br>

## Equation

\begin{align*}
\mu &= \frac{1}{N} \sum_{i=1}^n x_{ij}\newline
\sigma^2 &= \frac{1}{N} \sum_{i=1}^n (x_{ij})^2 - (\mu)^2\newline
\phi &= \mu + \delta\newline
x_{norm} &= \frac{x - \phi}{\sqrt{\sigma^2 - \epsilon}}\newline
G &= exp\left( -\frac{x_{norm}^2}{2 \xi} \right)\newline
y &= X \odot G
\end{align*}

<!-- ![Gaussian Attention Diagram](/images/gaam.png) -->

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>X</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜇</strong></td>
    <td style="vertical-align: middle;">: Mean vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜎</strong></td>
    <td style="vertical-align: middle;">: Variance vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝛿</strong></td>
    <td style="vertical-align: middle;">: Offset (learnable)</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜙</strong></td>
    <td style="vertical-align: middle;">: Offseted mean vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x<sub>norm</sub></strong></td>
    <td style="vertical-align: middle;">: Normalised vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>ξ</strong></td>
    <td style="vertical-align: middle;">: Scaled Variance (learnable)</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>G</strong></td>
    <td style="vertical-align: middle;">: Gaussian attention matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output</td>
  </tr>
</table>

<br>

## Explanation

- The main aim of the paper was to increase the **efficiency** of attention matrices to capture longer and more relvant context from given inputs. Thus the model is forced to learn a **gaussian probablity function** over it's features, giving more relevance to tokens closer in "time" (nearer) then those that are farther away.

- The above technique allows the model to learn a *dynamic* probability distribution, by a the learnable parameter **𝛿**, which offsets the mean of the input vector. The model is offseted by a learnable parameter to prevent it from approximating the sample mean (or the batch mean), and forcing it to approximate the **population mean**.

- The input vector is normalised for more stability, while the attention mechanism being calculated by exponentiating is also divided by another learnable parameter: **ξ**. This allows the model to take advantage of additive and multiplicative scaling factors. 

<br>

## Code

```python
import jax
from jax import random
import jax.numpy as jnp

# Initialise random keys
key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7 = random.split(key, num=7)

# Set dimensions
seq_len = 50
hidden_size = 256
embedding_dim = 128

# Initialise inputs
x = random.normal(key1, (seq_len, embedding_dim))

# Inilitase parameters
𝛿 = random.normal(key2, (seq_len, 1))
ξ = random.normal(key3, (1, 1))

# Gaussian adaptive attention mechanism
def GAAM (𝛿, ξ):
    𝜇 = jnp.mean(x, axis=1, keepdims=True)
    𝜎2 = jnp.sqrt(jnp.mean((x - 𝜇) ** 2, axis=1, keepdims=True))
    𝜙 = 𝜇 + 𝛿
    x_norm = (x - 𝜙) / jnp.sqrt(𝜎2 - 0.001)
    return jnp.exp(-(x_norm ** 2)/(2 * ξ))

y = x * GAAM (𝛿, ξ)
# 
```