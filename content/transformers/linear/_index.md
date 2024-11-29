---
title: "Linear Transformers"
date: 2024-11-26
draft: false
references:
  - title: "Autoregressive transformers with linear attention"
    url: "https://arxiv.org/abs/2006.16236"  
---

The vanilla transformer, or the dense transformer has a glaring flaw: it is ridiculously slow (with a quadratic complexity that increases with the sequence length). This is due to the inclusion of *every* token in the input for calculating the attention score, which would output a dense *n x n* matrix. As *n* increases, the square attention matrix and it's associated operations grow quadratically. This makes it infeasible to work with in Computer Vision and NLP tasks, hence faster alternatives were proposed.

<br>

## Equation

![Linear Attention Diagram](/images/lattention.png)

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
    <td style="vertical-align: middle;">: Query matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>K</strong><i></i></td>
    <td style="vertical-align: middle;">: Key matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>V</strong><i></i></td>
    <td style="vertical-align: middle;">: Value matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>A</strong><i></i></td>
    <td style="vertical-align: middle;">: Attention matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜙</strong><i></i></td>
    <td style="vertical-align: middle;">: Elu activation function</td>
  </tr>
</table>

<br>

## Explanation

- As mentioned before, dense attention became infeasible as the dataset sizes grew larger and larger, while the complexity was somewhat mititgated by using parallelisation, it was not enough. Hence various variants of transformers where proposed, which could work just as well, but more efficiently.

- Linear attention replaces the costly softmax operation with a simpler **elu activation** function, and *linearize* the calculation with reusing certain variables in the equation. This brings down the cost from *O(n^2)* to *O(n)*.

<br>

## Jax Code

```python
import jax
from jax import random
import jax.numpy as jnp

# Initialise random keys
key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8 = random.split(key, num=8)

# Set dimensions
batch_size=32
seq_len = 50
hidden_size = 256
embedding_dim = 128

# Initialise inputs
x = random.normal(key1, (batch_size, seq_len, embedding_dim))

# Inilitase parameters
W_q = random.normal(key2, (embedding_dim, hidden_size))
W_k = random.normal(key3, (embedding_dim, hidden_size))
W_v = random.normal(key4, (embedding_dim, hidden_size))
W_O = random.normal(key4, (hidden_size, embedding_dim))
b_q = random.normal(key5, (hidden_size,))
b_k = random.normal(key6, (hidden_size,))
b_v = random.normal(key7, (hidden_size,))
b_o = random.normal(key8, (embedding_dim,))

def LinearAttention (W_q, W_k, W_v, x):
  
  Q = jax.nn.elu(jnp.dot(x, W_q) + b_q) + 1
  K = jax.nn.elu(jnp.dot(x, W_k) + b_k) + 1
  V = jnp.dot(x, W_v) + b_v
    
  # KV matrix calculation
  KV = jnp.einsum("nsd,nsm->nmd", K, V)

  # normalizing state
  Z = 1/(jnp.einsum("nld,nd->nl", Q, K.sum(axis=1)))
  V = jnp.einsum("nld,nmd,nl->nlm", Q, KV, Z)

  return V

# Project the attention scores to output
A = LinearAttention(W_q, W_k, W_v, x)
y = jnp.dot(A, W_O) + b_o
```