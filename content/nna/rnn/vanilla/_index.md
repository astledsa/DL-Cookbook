---
title: "Recurrent Neural Network"
date: 2023-04-20
draft: false
references:
    - title: "RNN overview"
      url: "https://arxiv.org/abs/1912.05911"
---

The bedrock of compressed contextual understanding and one of the only methods for early natural language processing tasks, RNNs were some of the best architectures until LSTMs and transformers took over. Some people still argue about the effectiveness of RNNs, and we still see a few [papers](https://arxiv.org/abs/2410.01201) on them ounce in a while.

<br>

## Equation

![MLP Diagram](/images/rnn2.png)

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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>H</strong><sub><i>t</i></sub></td>
    <td style="vertical-align: middle;">: Hidden state at time <i>t</i></td>
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

- Natural language, when viewed from a statistical point of view, has two important properties that a model architecture must take into consideration: **sequentiality** and **markovian** assumption. The ordered form of the data, and the assumption that they are dependant on the sample that came before it were two inductive biases that the model must assume.

- Both of the above properties are taken care of by the passage of **hidden states**, which captures past information and relevant context (making sure the next state is dependant on the previous ones). The nature of the flow of the gradients, which is sequentially backwards, also known as **backpropagation through time or BPTT**, allows for the learned information to flow to the earlier hidden states, and hence ensures the depedancy or the markovian assumption, where this learned information can help the model make better informed decisions to predict the next states.

<br>

## Code

```python
import jax 
from jax import random
import jax.numpy as jnp

key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6 = random.split(key, num=6)

# The dimension of our hidden layer
hidden_size = 256

# A random vector as an input
x = random.normal(key1, (50, 128))
H = jnp.zeros((50, hidden_size))

# Initialise the weights or parameters the required dimensions
W_xh = random.normal(key2, (128, hidden_size))
W_hh = random.normal(key3, (hidden_size, hidden_size))
W_ho = random.normal(key4, (hidden_size, 128))
b_t = random.normal(key5, (1, hidden_size))
b_o = random.normal(key6, (1, 128))

# Ht = 𝜙(XW_xh + Ht-1W_hh + bt)
def first_layer (x, H, W_xh, W_hh, b_t):
  return jax.nn.relu(jnp.dot(x, W_xh) + jnp.dot(H, W_hh) + b_t)

# O = HtW_ho + bo
def final_layer (ht, W_ho, b_o):
  return jnp.dot(ht, W_ho) + b_o

H = first_layer(x, H, W_xh, W_hh, b_t)
y = final_layer(H, W_ho, b_o)
```