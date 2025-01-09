---
title: "Gated Recurrent Unit"
date: 2023-04-20
draft: false
references:
    - title: "Long short-term memory [PDF]"
      url: "https://www.bioinf.jku.at/publications/older/2604.pdf"
---

A simple modification over RNNs and LSTMs was the gated recurrent unit, made to retain the features of LSTMs but also aid in faster computation, as LSTMs were quite expensive in larger networks

<br>

## Equation

![MLP Diagram](/images/gru2.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Hidden State</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Weight Matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Bias vectors</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>R</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Reset gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>Z</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Update gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H~</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Candidate hidden state</td>
  </tr>
</table>

<br>

## Explanation

- As mentioned before, the GRU was mainly constructed to reduce the amount of gated mechanisms used, since each gate requires a weight matrix to be learned, making LSTMs quite costly in longer and wider networks. The three gates were replaced by two: **Reset** and **Update** gate.

- The reset gate controls how much of the previous state is to be retained/remembered, and the update gate controls how much the new state is a copy of the old state. The reset gate can either make sure the entirety of previous state's information is passed forward, or none of it (both ends thus belong to RNNs and MLPs respectively). The same goes for the update gate. If the value of the gate is one, than the entire old input is retained, and vice versa. (The gate's values are constrianed between 0 and 1 through a sigmoid function).

<br>

## Code

```Python
import jax 
from jax import random
import jax.numpy as jnp

key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8 = random.split(key, num=8)

# The dimension of our hidden layer
hidden_size = 256

# A random vector as an input
x = random.normal(key1, (50, 128))
H = jnp.zeros((50, hidden_size))

# Initialise the weights or parameters the required dimensions
W_xr = random.normal(key2, (128, hidden_size))
W_hr = random.normal(key3, (hidden_size, hidden_size))
W_xz = random.normal(key4, (128, hidden_size))
W_hz = random.normal(key5, (hidden_size, hidden_size))
W_xh = random.normal(key6, (128, hidden_size))
W_hh = random.normal(key7, (hidden_size, hidden_size))
W_ho = random.normal(key8, (hidden_size, 128))
b_r = random.normal(key2, (1, hidden_size))
b_z = random.normal(key4, (1, hidden_size))
b_h = random.normal(key6, (1, hidden_size))
b_o = random.normal(key8, (1, 128))

def gru_cell(x, H, W_xr, W_hr, W_xz, W_hz, W_xh, W_hh, b_r, b_z, b_h):
    r = jax.nn.sigmoid(jnp.dot(x, W_xr) + jnp.dot(H, W_hr) + b_r)
    z = jax.nn.sigmoid(jnp.dot(x, W_xz) + jnp.dot(H, W_hz) + b_z)
    h_tilde = jax.nn.tanh(jnp.dot(x, W_xh) + jnp.dot(r * H, W_hh) + b_h)
    H_next = (1 - z) * H + z * h_tilde
    return H_next

# 10 iterations or GRU layers
for _ in range(10):
    H = gru_cell(x, H, W_xr, W_hr, W_xz, W_hz, W_xh, W_hh, b_r, b_z, b_h)

y = jnp.dot(H, W_ho) + b_o
```