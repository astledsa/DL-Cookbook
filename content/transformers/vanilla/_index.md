---
title: "Vanilla/Dense Transformers"
date: 2024-11-26
draft: false
references:
  - title: "Attenion in Neural translation"
    url: "https://arxiv.org/pdf/1409.0473"
  - title: "Attention is all you need"
    url : "https://arxiv.org/abs/1706.03762"
  - title: "Language Models are few-shot learners"
    url: "https://arxiv.org/abs/2005.14165"
  - title: "Geometry of concepts in LLMs"
    url: "https://arxiv.org/html/2406.01506v1#:~:text=We%20find%20a%20remarkably%20simple,simplices%2C%20reflecting%20the%20hierarchical%20structure."
  - title: "A mathematical framework for transformers"
    url: "https://transformer-circuits.pub/2021/framework/index.html#additional-intuition"
  - title: "In-context learning and Induction heads"
    url: "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html"
  - title: "Chain of Thought empowers transformers to solve serial problems"
    url: "https://arxiv.org/abs/2402.12875"      
---

The advent of transformers, alongside massive scaling revitalized the field of deep learning. "AI" became a thing that was discussed by not just enthusiasts and theorists but by every normal person ounce again. This architecture is the backbone of the "AI boom" we are currently (as of this writing - 2024) experiencing.

<br>

## Equation

![Dense Attention Diagram](/images/dattention.png)

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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>A</strong><i></i></td>
    <td style="vertical-align: middle;">: Attention matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>V</strong><i></i></td>
    <td style="vertical-align: middle;">: Value matrix</td>
  </tr>
</table>

<br>

## Explanation

- There have been several theoretical frameworks proposed to understand the inner workings of the attention mechanism, or how transformers learn (and *why* they are so good at it). One of the simplest approaches is to view it as a simple retrieval task: give a query vector, find the most similar key vector, and return it's corresponding value vector as the final answer. The way these vectors are weighted is propotional to all the other vectors, before *and* after the current token (so it also accounts for future tokens, exactly like in bidirectionl RNNs), and produces a weighted sum of it. This case suits perfectly for NLP tasks, but also worked really well with image-related tasks (ViTs).

- A very powerful, and indeed one of the most important features of transformers, is it's ability to learn from any given context, or in-context learning. This enabled it to learn to perform any given NLP task, given a sufficient prompt [3]. Various other theories exist [4, 5, 6, 7], which I shall refrain from mentioning here, as the theoretical framework is yet to be solidified.

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
W_q = random.normal(key2, (embedding_dim, hidden_size))
W_k = random.normal(key3, (embedding_dim, hidden_size))
W_v = random.normal(key4, (embedding_dim, hidden_size))
W_O = random.normal(key4, (hidden_size, embedding_dim))
b_q = random.normal(key5, (hidden_size,))
b_k = random.normal(key6, (hidden_size,))
b_v = random.normal(key7, (hidden_size,))
b_o = random.normal(key7, (embedding_dim,))

def DenseAttention (W_q, W_k, W_v, x):
    Q = jnp.dot(x, W_q) + b_q
    K = jnp.dot(x, W_k) + b_k
    V = jnp.dot(x, W_v) + b_v
    
    return jax.nn.softmax((Q @ jnp.transpose(K)) / jnp.sqrt(hidden_size), axis=-1) @ V

# Project the attention scores to output
A = DenseAttention(W_q, W_k, W_v, x)
y = jnp.dot(A, W_O) + b_o
```