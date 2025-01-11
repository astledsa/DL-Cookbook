---
title: "Differential Transformers"
date: 2024-12-10
draft: false
references:
  - title: "Differential Transformers"
    url: "https://arxiv.org/abs/2410.05258"  
---

The problems that needed to be solved have changed since the advent of deep learning. From focusing on stable training to even making a coherent language model, it has shifted to increasing the context length of an LLM. This shift has led the next wave of variants of transformers, while some looking to increase it's context length, and others focusing on making them more efficient.

<br>

## Equation

![Differential Attention Diagram](/images/diff.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong><i></i></td>
    <td style="vertical-align: middle;">: Weight matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b</strong></td>
    <td style="vertical-align: middle;">: Bias vectors</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>Q</strong><i></i></td>
    <td style="vertical-align: middle;">: Query matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>K</strong><i></i></td>
    <td style="vertical-align: middle;">: Key matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>A</strong><i></i></td>
    <td style="vertical-align: middle;">: Attention matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>V</strong><i></i></td>
    <td style="vertical-align: middle;">: Value matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜆</strong><i></i></td>
    <td style="vertical-align: middle;">: learnable scalar</td>
  </tr>
</table>

<br>

## Explanation

- The attention mechanism essentially captutes *noise* as well, while calculating the attention scores of a given context. This addition of unnecessary context/noise may hinder the accuracy of the final token prediction (or make it less relevant/hallucinate). This problem was mitigated by a simple operation: **differencing the softmax outputs**.

- The purpose of differencing the softmax outputs was that it **cancels out the noise**, and essentially helps the attention matrices focus only on the relevant context. This simple modification (as shown in the paper), improves the perfomance of transformers in language models.

<br>

## Code

```python
import jax
from jax import random
import jax.numpy as jnp

# Initialise random keys
key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13 = random.split(key, num=13)

# Set dimensions
seq_len = 50
hidden_size = 256
embedding_dim = 128

# Initialise inputs
x = random.normal(key1, (seq_len, embedding_dim))

# Inilitase parameters
W_q = random.normal(key2, (embedding_dim, 2 * hidden_size))
W_k = random.normal(key3, (embedding_dim, 2 * hidden_size))
W_v = random.normal(key4, (embedding_dim, hidden_size))
W_O = random.normal(key4, (hidden_size, embedding_dim))
b_q = random.normal(key5, (2 * hidden_size,))
b_k = random.normal(key6, (2 * hidden_size,))
b_v = random.normal(key7, (hidden_size,))
b_o = random.normal(key8, (embedding_dim,))
𝜆_q1 = random.normal(key9, (1, hidden_size))
𝜆_q2 = random.normal(key10, (1, hidden_size))
𝜆_k1 = random.normal(key11, (1, hidden_size))
𝜆_k2 = random.normal(key12, (1, hidden_size))
𝜆_init =  random.normal(key13, (1,1)) 

def DiffAttention (W_q, W_k, W_v, x):
    Q1, Q2 = jnp.split(jnp.dot(x, W_q) + b_q, 2, axis=1)
    K1, K2 = jnp.split(jnp.dot(x, W_k) + b_k, 2, axis=1)
    V = jnp.dot(x, W_v) + b_v
    A1 = jax.nn.softmax((Q1 @ jnp.transpose(K1)) / jnp.sqrt(hidden_size), axis=-1)
    A2 = jax.nn.softmax((Q2 @ jnp.transpose(K2)) / jnp.sqrt(hidden_size), axis=-1)

    𝜆 = jnp.exp(jnp.dot(𝜆_q1, 𝜆_k1.T)) - jnp.exp(jnp.dot(𝜆_q2, 𝜆_k2.T)) + 𝜆_init
    return (A2 - (𝜆 * A1)) @ V


# Project the attention scores to output
A = DiffAttention(W_q, W_k, W_v, x)
y = jnp.dot(A, W_O) + b_o
#
```