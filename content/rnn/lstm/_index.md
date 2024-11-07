---
title: "Long Short-Term Memory"
date: 2023-04-20
draft: false
references:
    - title: "Long short-term memory [PDF]"
      url: "https://www.bioinf.jku.at/publications/older/2604.pdf"
---

One of the most used and known variants is the LSTM: long short-term memory network. This particular variant was originally created to help mitigate the fundamental problem of exploding and vanishing gradients in long sequence modelling, and hence had been the standard for over a decade.

<br>

## Equation

![MLP Diagram](/images/lstm.png)

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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>I</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Input gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>F</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Forget gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>O</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Output gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>C</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Cell State</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>C~</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Input Node</td>
  </tr>
</table>

<br>

## Explanation

- All variants (including this one) of a model have the same inductive biases as their parent architecture. LSTMs were invented in order to mitigate the gradient exploding/vanishing problem by essentially *constraining* the flow of information from the input during forward pass, and the flow of error during the backward pass.

- The **Input** gate handles how much information from the new input must be retained (given it's importance during loss minimization) through the input *node*, which is each layer's (called a *cell* in academic papers) internal state. The **Forget** gate does exactly what it's name suggests: helps the network forget irrelavant information from the input.

- Finally, the **Output** gate decides whether the current cell state should affect other layers or not. The **Cell state** represents flow of information throughout the layer, which is *controlled* by the gates in order to control the flow of information (from the input) and error (loss gradients) from exponentially growing and decaying. This (somewhat) mitigates the need for [gradient clipping](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48) and solves the vanishing gradient problem.

<br>

## Jax Code

```python
import jax 
from jax import random
import jax.numpy as jnp

key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8, key9, key10 = random.split(key, num=10)

# The dimension of our hidden layer
hidden_size = 256

# A random vector as an input
x = random.normal(key1, (50, 128))
H = jnp.zeros((50, hidden_size))
C = jnp.zeros((50, hidden_size))

# Initialise the weights or parameters the required dimensions
W_xi = random.normal(key2, (128, hidden_size))
W_hi = random.normal(key3, (hidden_size, hidden_size))
W_xf = random.normal(key4, (128, hidden_size))
W_hf = random.normal(key5, (hidden_size, hidden_size))
W_xc = random.normal(key6, (128, hidden_size))
W_hc = random.normal(key7, (hidden_size, hidden_size))
W_xo = random.normal(key8, (128, hidden_size))
W_ho = random.normal(key9, (hidden_size, hidden_size))
W_ho = random.normal(key10, (hidden_size, 128))
b_i = random.normal(key2, (1, hidden_size))
b_f = random.normal(key4, (1, hidden_size))
b_c = random.normal(key6, (1, hidden_size))
b_o = random.normal(key8, (1, hidden_size))
b_y = random.normal(key10, (1, 128))

def lstm_cell(x, H, C, W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho, b_i, b_f, b_c, b_o):
    i = jax.nn.sigmoid(jnp.dot(x, W_xi) + jnp.dot(H, W_hi) + b_i)
    f = jax.nn.sigmoid(jnp.dot(x, W_xf) + jnp.dot(H, W_hf) + b_f)
    c = jax.nn.tanh(jnp.dot(x, W_xc) + jnp.dot(H, W_hc) + b_c)
    o = jax.nn.sigmoid(jnp.dot(x, W_xo) + jnp.dot(H, W_ho) + b_o)
    C_next = f * C + i * c
    H_next = o * jax.nn.tanh(C_next)
    return H_next, C_next

for _ in range(10):
    H, C = lstm_cell(x, H, C,
                     W_xi, W_hi, W_xf, W_hf, W_xc, W_hc, W_xo, W_ho, 
                     b_i, b_f, b_c, b_o
                    )

y = jnp.dot(H, W_ho) + b_y
```