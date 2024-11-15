---
title: "Bidirectional Recurrent Neural Network"
date: 2023-04-20
draft: false
references:
    - title: "Bidirectional Recurrent Neural Networks [PDF]"
      url: "https://deeplearning.cs.cmu.edu/F20/document/readings/Bidirectional%20Recurrent%20Neural%20Networks.pdf"
---

Here is one of the first variants of recurrent neural networks, first introduced in 1997, which seemed to cover for one of the basic limitations of RNNs : not having *future* information to predict an output, which could be valuable as well.

<br>

## Equation

![MLP Diagram](/images/brnn.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong><i><sub>xh</sub></i></td>
    <td style="vertical-align: middle;">: Weight matrix for input</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong><i><sub>hh</sub></i></td>
    <td style="vertical-align: middle;">: Weight matrix for hidden state</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W</strong><i><sub>ho</sub></i></td>
    <td style="vertical-align: middle;">: Weight matrix for output</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b</strong><sub><i>t</i></sub></td>
    <td style="vertical-align: middle;">: Bias for input/hidden states</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b</strong><sub><i>o</i></sub></td>
    <td style="vertical-align: middle;">: Bias for output</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H</strong><sub><i>bt</i></sub></td>
    <td style="vertical-align: middle;">: backward Hidden state at time <i>t</i></td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H</strong><sub><i>ft</i></sub></td>
    <td style="vertical-align: middle;">: forward  Hidden state at time <i>t</i></td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>O</strong></td>
    <td style="vertical-align: middle;">: Output matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>t</i></td>
    <td style="vertical-align: middle;">: Temporal Index</td>
  </tr>
</table>

<br>

## Explanation

- A simple variant of the RNN architecture, this variant is based on a simple assumption that the next token to be predicted can (and often does) depend on the *future* tokens, or the context that comes *after* the current time as well, and hence even that must be taken into account for a better probablity of a more accurate prediction.

- The procedure/forward pass is a simple modification over the vanilla RNN, where the hidden state has two parts: one that contains information from the "past" and one that has information from the "future". These states are than concatenated before being passed into the final layer.

<br>

## Jax Code

```python
import jax 
from jax import random
import jax.numpy as jnp

key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8 = random.split(key, num=8)

# The dimension of our hidden layer
hidden_size = 256

# A random vector as an input
x = random.normal(key1, (50, 128))
H_forward = jnp.zeros((50, hidden_size))
H_backward = jnp.zeros((50, hidden_size))

# Initialise the weights or parameters the required dimensions
W_xh_forward = random.normal(key2, (128, hidden_size))
W_hh_forward = random.normal(key3, (hidden_size, hidden_size))
W_xh_backward = random.normal(key4, (128, hidden_size))
W_hh_backward = random.normal(key5, (hidden_size, hidden_size))
W_ho = random.normal(key6, (2 * hidden_size, 128))
b_t_forward = random.normal(key7, (1, hidden_size))
b_t_backward = random.normal(key8, (1, hidden_size))
b_o = random.normal(key5, (1, 128))

# Forward pass
def forward_layer(x, H, W_xh, W_hh, b_t):
  return jax.nn.relu(jnp.dot(x, W_xh) + jnp.dot(H, W_hh) + b_t)

# Backward pass
def backward_layer(x, H, W_xh, W_hh, b_t):
  return jax.nn.relu(jnp.dot(x, W_xh) + jnp.dot(H, W_hh) + b_t)

# Output layer
def final_layer(h_forward, h_backward, W_ho, b_o):
  h_concat = jnp.concatenate([h_forward, h_backward], axis=1)
  return jnp.dot(h_concat, W_ho) + b_o

H_forward = forward_layer(x, H_forward, W_xh_forward, W_hh_forward, b_t_forward)
H_backward = backward_layer(jnp.flip(x, axis=0), H_backward, 
                            W_xh_backward, W_hh_backward, b_t_backward
                            )
y = final_layer(H_forward, H_backward, W_ho, b_o)
```