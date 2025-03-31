---
title: "Universal Transformers"
date: 2024-12-10
draft: false
references:
  - title: "Universal Transformers"
    url: "https://arxiv.org/abs/1807.03819"  
---

An older (comparatively) version of transformers, this was an early attempt at improving upon the transformer architecture, by imbuing it with features of recurrent networks. This resulted in a recurrent-like transformer, which was named as the Universal transformer (since the authors saw this as a generalization of the transformer). In the paper they have applied it to the encoder-decoder architecture, but here I assume it to be a decoder-only model, similiar to modern LLMs.

<br>

## Equation

\begin{align}
Q &= W_q (H_{t-1} + P_t) + b_q\newline
K &= W_k (H_{t-1} + P_t) + b_k\newline
V &= W_v (H_{t-1} + P_t) + b_v\newline
O &= W_o \left( softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) V \right) + b_o\newline
A_t &= L(H_{t-1} + P_t) + O\newline
H_t &= L(A_t + T(A_t))
\end{align}

<!-- ![Universal Attention Diagram](/images/ut.png) -->

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H</strong><sub><i>t</i></sub></td>
    <td style="vertical-align: middle;">: Hidden Representatoins</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>P</strong><sub><i>t</i></sub></td>
    <td style="vertical-align: middle;">: Positional Embedding matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong></td>
    <td style="vertical-align: middle;">: Weight Matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b</strong></td>
    <td style="vertical-align: middle;">: Bias Vectors</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>O</strong><sub></td>
    <td style="vertical-align: middle;">: Attention-output matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>A</strong><sub><i>t</i></sub></td>
    <td style="vertical-align: middle;">: Skip-Connected Matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>L()</i><sub></td>
    <td style="vertical-align: middle;">: LayerNorm function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>T()</i><sub></td>
    <td style="vertical-align: middle;">: Transition function</td>
  </tr>
</table>

<br>

## Explanation

- Universal transformers can, in simple words, be seen as an **amalagation** of recurrent neural networks and the transformer network. The theory behind this was that the recurrent nature of RNNs allowed them some capabalities that transformers simply could not replicate, even though the tasks were relatively simple. On the other hand, recurrent networks suffer from the bottleneck of being inherently sequential, hence parallelizing them is non-trivial. Thus combining the best of both worlds, the authors suggest a recurrent-natured transformer architecture, and called it the Universal Transformer (since it can be seen as a generalization of the vanilla transformer)

- UT performs the same operations as the attention mechanism, apart from it's maintanance of a **hidden state represenatation**, and the **transition function**. Unlike recurrent networks which perform recurrent operations over the sequence, the UT performs recurrent operations over the vector representations of the tokens (which are informed with the information of other tokens thanks to the attention mechanism), and can do so a variable number of times. This means the networks recurrently processes over the "depth" of the token representations. This gives the UT **variable depth**.

- The authors have also implemented the **Adaptive Compute Time Halting mechanism**. This mechanism gives the model control over how much recurrent processing to perform on each token (since, according to the authors, each token has different levels of ambiguity, and more difficult words should require more "thinking" time), hence the variable depth of the UTs is complimented by an ability to **dynamically** control _how much_ compute to spend on each token.

- The intuition is that we humans do not give the same amount of thinking time to each and every word in a sentence: we "**ponder**" more on difficult/confusing/complex words, and hence have a dynamic thinking process, as opposed to most neural networks: which give the same amount of compute power (analogous to a human _thinking_) to each token in the sequence. Hence with the combination of **UT** alongside the **ACT**, the authors seek to mimic the human thinking process (a common theme in Artifical _Intelligence_).

<br>

## Code

```python
import jax
from jax import lax
from jax import random
import flax.linen as nn
import jax.numpy as jnp

step = 1
max_steps = 2
threshold = 0.5
seq_len = 100
hidden_size = 256
embedding_dim = 128
key = random.PRNGKey(100)
key_parts = random.split(key, 12)

n_updates = jnp.zeros(seq_len)
remainders = jnp.zeros(seq_len)
halting_probability = jnp.zeros(seq_len)
previous = jnp.zeros((seq_len, embedding_dim))
x = random.normal(key_parts[0], (seq_len, embedding_dim))

weights = {
    'b_q': random.normal(key_parts[0], (hidden_size,)),
    'b_k': random.normal(key_parts[1], (hidden_size,)),
    'b_v': random.normal(key_parts[2], (hidden_size,)),
    'b_o': random.normal(key_parts[3], (embedding_dim,)),
    'W_q': random.normal(key_parts[4], (embedding_dim, hidden_size)),
    'W_k': random.normal(key_parts[5], (embedding_dim, hidden_size)),
    'W_v': random.normal(key_parts[6], (embedding_dim, hidden_size)),
    'W_o': random.normal(key_parts[7], (hidden_size, embedding_dim)),
    'W1': random.normal(key_parts[8], (hidden_size, hidden_size)),
    'b1': random.normal(key_parts[9], (hidden_size,)),
    'W2': random.normal(key_parts[10], (hidden_size, embedding_dim)),
    'b2': random.normal(key_parts[11], (embedding_dim,)), 
    'W': random.normal(key, (embedding_dim,)), 
    'b': random.normal(key, (seq_len,)),
}

# Determine when to stop the recursive loop
def should_continue(state):
    (_, _, halting_probability, _, n_updates, _, _) = state
    condition = (halting_probability < threshold) & (n_updates < max_steps)
    return jnp.any(condition).astype(jnp.bool_)

# Defining the postional and time coordinate matrix
def P (m, d, t):
    x = random.normal(random.PRNGKey(t), (m, d))
    for i in range(m):
        for j in range(0, d, 2):
            idx = int(j // 2)
            x = x.at[i, j].set(jnp.sin(i / (10_000**idx)) + jnp.sin(1 / (10_000**idx)))
            x = x.at[i, (j+1) % d].set(jnp.cos(i / (10_000**idx)) + jnp.cos(1 / (10_000**idx)))
    return x

def SelfAttention (x, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o) :
  Q = jnp.dot(x, W_q) + b_q
  K = jnp.dot(x, W_k) + b_k
  V = jnp.dot(x, W_v) + b_v
  return jax.nn.softmax((Q @ jnp.transpose(K)) / jnp.sqrt(hidden_size), axis=-1) @ V

# The transition function to be applied on the state
def Transition (A_t, W1, b1, W2, b2) :
  A_t1 = jnp.dot(A_t, W1) + b1
  A_t2 = jnp.maximum(0, A_t1)
  A_t_out = jnp.dot(A_t2, W2) + b2
  return A_t_out

# The Adaptive dynamic halting mechanism. It ensures the model
# runs on a variable depth on each token, taking a non-uniform
# amount of time on processing each token
def DynamicHalting (S) :
  (state, step, halting_probability, remainders, n_updates, previous_state, weights) = S
  state += P(seq_len, embedding_dim, step)
  p = jax.nn.sigmoid(jax.nn.relu(jnp.dot(state, weights['W']) + weights['b']))
  still_running = jnp.float32(jnp.array(halting_probability < 1))
  new_halted = jnp.float32(
      jnp.array(
          halting_probability + p * still_running > threshold
        )) * still_running
  
  still_running = jnp.float32(
      jnp.array(
          halting_probability + p * still_running <= threshold
        )) * still_running
   
  halting_probability += p * still_running
  remainders += new_halted * (1 - halting_probability)
  halting_probability += new_halted * remainders
  n_updates += still_running + new_halted
  update_weights = jnp.expand_dims(
      p * still_running + new_halted * remainders, -1
  )
   
  transformed_state = Transition(
      SelfAttention(
          state, weights['W_q'], weights['W_k'], weights['W_v'], 
          weights['W_o'], weights['b_q'], weights['b_k'], weights['b_v'], 
          weights['b_o']
        ), weights['W1'], weights['b1'], weights['W2'], weights['b2']
      )
  new_state = (
      (transformed_state * update_weights) + 
       (previous_state * (1 - update_weights))
      )
  step += 1
  return ( transformed_state, step, halting_probability, 
          remainders, n_updates, new_state, weights )

state = (x, step, halting_probability, remainders, n_updates, previous, weights)
while should_continue(state):
  state = DynamicHalting(state)

(_, _, _, remainders, n_updates, new_state, weights) = state
```
*The above code is more or less a copy from the original paper, only I have converted it to be a simple decoder only block and written it in Jax. Kindly refer to the paper for a line-by-line explanation of the Dynamic halting function*