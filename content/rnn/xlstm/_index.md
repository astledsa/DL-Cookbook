---
title: "Extended Long Short-Term Memory"
date: 2023-04-20
draft: false
references:
    - title: "xLSTM: Extended Long short-term memory"
      url: "https://arxiv.org/abs/2405.04517"
---

In the modern era of Transformers, LSTMs were long forgotten, until now. The following was an experiment to see whether mitigating the shortcomings of LSTMs, and scaling LSTM inspired LLMs can actually rival transformers in processing longer-contexts and modelling language itself.

<br>

## sLSTM Equation

![MLP Diagram](/images/slstm.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>X</strong></td>
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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>N</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Normalizer State</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>Z</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Cell Input</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>M</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Stabalizer State</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>I'</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Stabalized Input gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>F'</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Stabalized Forget gate</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜎</strong></td>
    <td style="vertical-align: middle;">: Sigmoid function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>φ</strong></td>
    <td style="vertical-align: middle;">: TanH function</td>
  </tr>
</table>

<br>

## mLSTM Equation

![MLP Diagram](/images/mlstm.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>X</strong></td>
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
    <td style="padding-right: 20px; vertical-align: middle;"><strong>Q</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Query Matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>K</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Key Matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>V</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Value Matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>N</strong><sub>t</sub></td>
    <td style="vertical-align: middle;">: Normalizer State</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜎</strong></td>
    <td style="vertical-align: middle;">: Sigmoid function</td>
  </tr>

</table>

<br>

## Explanation

- LSTMs had certain shortcomings, which were the reasons it got replaced by transformers in recent time. Three such limitations can be identified: i)  lack of revising storage decisions, or in other words, accurate **context retrieval**, ii) **limited storage capabalities**, since just like RNNs, even LSTMs are forced to compress all of the past information into the hidden state, and iii) they are **inherently sequential**, and hence parallelization is a non-trivial problem.

- This particular architecture aims to mitigate the above problems, and hence truly compare LSTM insipred Langauge models and their limitations and capabalities. The two modifications are **sLSTM** blocks and **mLSTM** blocks, which can be used together to create the overall xLSTM architecture.

- sLSTM tries to deal with the problem of revising storage decision by replacing the sigmoid function of the input and forget gates with the **exponential** activation function. The exponential function essentially _enhances_ the expressive power of the respective gates, hence the input gate gets more dynamic control over what new information enters the cell, while the forget gate also gets more expressive, allowing for more dynamic forgetting and retaining of old information. (_"Dynamic" here means the gates can make more drastic changes to the flow of information_)

- The storage problem was mitigated by **incrementing the dimension of the cell state by one** (in the paper they replaced the scalar by a matrix, **C**, but since we are dealing with input vectors in batches, we have to account for the increased dimensions). This along with the **removal of dependancy on the previous hidden state** allowed it to be parallelizable (just like transformers), _and_ have more information storage capabalities. The usage of mechanisms similar to transformers (query, key and value vectors) aids in better context retrieval as well.

- These two blocks together constitute the overall xLSTM architecture. The appropriate number of each block can lead to increase in model perfomance (it's not imperative that both blocks should be equal, but other factors like sLSTM not being paralellizable, must be considered). Another implementation detail is that the inputs are first projected into higher dimensions, where it is processed (similar to transformers) by mLSTM blocks, while in sLSTM's case, the input is processed first before projection (like in SSMs).

<br>

## Jax Code

```python
import jax 
from jax import random
import jax.numpy as jnp

def causal_conv1d(x, kernel):
  
  x = jnp.transpose(x, (0, 2, 1))
  x = jnp.pad(x, ((0, 0), (0, 0), (3, 0)))
  output = jax.lax.conv_general_dilated(x, kernel,
                                       window_strides=(1, ),
                                       padding='VALID')
  output = jnp.transpose(output, (0, 2, 1))

  return output


def layer_norm(X, epsilon=1e-5):

  mean = jnp.mean(X, axis=-1, keepdims=True)
  variance = jnp.var(X, axis=-1, keepdims=True)
  X_normalized = (X - mean) / jnp.sqrt(variance + epsilon)

  return X_normalized


def group_norm(X, num_groups=2, epsilon=1e-5):

  batch_size, length, channels = X.shape
  assert channels % num_groups == 0, "The number of channels must be divisible by num_groups"
  group_size = channels // num_groups
  X_grouped = X.reshape(batch_size, length, num_groups, group_size)
  mean = jnp.mean(X_grouped, axis=(2, 3), keepdims=True)
  variance = jnp.var(X_grouped, axis=(2, 3), keepdims=True)
  X_grouped = (X_grouped - mean) / jnp.sqrt(variance + epsilon)
  X_normalized = X_grouped.reshape(batch_size, length, channels)
  
  return X_normalized


class sLSTM:

  def __init__ (self, input, hidden_size, 
                projection_factor=4/3, kernel_size=4, 
                key = random.PRNGKey(100)
              ):
    
    # Input Dimension: (Batch Size, Sequence Length, Embedding dimension)

    assert (input.ndim >= 2)

    if input.ndim == 2:
      input = input[jnp.newaxis, :, :]

    keys = {f"key{i+1}": k for i, k in enumerate(random.split(key, num=19))}
    self.input = input
    self.states = {
        "N": jnp.ones((input.shape[1], hidden_size)),
        "C": jnp.ones((input.shape[1], hidden_size)),
        "M": jnp.ones((input.shape[1], hidden_size)),
        "H": jnp.ones((input.shape[1], hidden_size))
    }

    self.params = {
        "W_xi": random.normal(keys["key1"], (input.shape[2], hidden_size)),
        "W_xf": random.normal(keys["key2"], (input.shape[2], hidden_size)),
        "W_xo": random.normal(keys["key3"], (input.shape[2], hidden_size)),
        "W_xz": random.normal(keys["key4"], (input.shape[2], hidden_size)),
        "W_ho": random.normal(keys["key5"], (hidden_size, input.shape[2])),
        "R_xi": random.normal(keys["key6"], (hidden_size, hidden_size)),
        "R_xf": random.normal(keys["key7"], (hidden_size, hidden_size)),
        "R_xo": random.normal(keys["key8"], (hidden_size, hidden_size)),
        "R_xz": random.normal(keys["key9"], (hidden_size, hidden_size)),
        "b_i": random.normal(keys["key10"], (1, hidden_size)),
        "b_f": random.normal(keys["key11"], (1, hidden_size)),
        "b_o": random.normal(keys["key12"], (1, hidden_size)),
        "b_z": random.normal(keys["key13"], (1, hidden_size)), 
        "b_y": random.normal(keys["key14"], (1, input.shape[2])),
        "W_dp": random.normal(keys["key15"], (int(input.shape[2] * projection_factor), 
                                              input.shape[2]
                                            )),
        "W_up_1": random.normal(keys["key17"], (int(input.shape[2] * projection_factor), 
                                              input.shape[2]
                                            )),
        "W_up_2": random.normal(keys["key17"], (int(input.shape[2] * projection_factor), 
                                              input.shape[2]
                                            )),
        "b_dp": random.normal(keys["key16"], (1, input.shape[2])),
        "b_up_1": random.normal(keys["key18"], (1, int(input.shape[2] * projection_factor))),
        "b_up_2": random.normal(keys["key18"], (1, int(input.shape[2] * projection_factor))),
        "kernel": random.normal(keys["key19"], (
          input.shape[-1], 
          input.shape[-1], 
          kernel_size
        ))
    }

  def cell (self, x, x_conv, 
                  H_old, N, C, M_old, 
                  W_xi, W_xf, W_xo, W_xz, 
                  R_xi, R_xf, R_xo, R_xz, 
                  b_i, b_f, b_o, b_z
                ):
      
    _H = C / N
    I = jnp.exp(jnp.dot(x_conv, W_xi) + jnp.dot(H_old, R_xi) + b_i)
    F = jnp.exp(jnp.dot(x_conv, W_xf) + jnp.dot(H_old, R_xf) + b_f)
    O = jax.nn.sigmoid(jnp.dot(x, W_xo) + jnp.dot(H_old, R_xo) + b_o)
    Z = jax.nn.tanh(jnp.dot(x, W_xz) + jnp.dot(H_old, R_xz) + b_z)
    C = F * C + I * Z
    N = F * N + I
    H = O * _H
    M = jnp.maximum(jnp.log(F) + M_old, jnp.log(I))
    I = jnp.exp(jnp.log(I) - M)
    F = jnp.exp(jnp.log(F) + M_old - M)

    return H, (C, N, M)

  def forward (self) :
      
    x = layer_norm(self.input)
    x_conv = jax.nn.swish(causal_conv1d(x, self.params["kernel"]))

    H, _ = self.cell(
        x, x_conv, 
        self.states["H"], self.states["N"], self.states["C"], self.states["M"], 
        self.params["W_xi"], self.params["W_xf"], self.params["W_xo"], self.params["W_xz"], 
        self.params["R_xi"], self.params["R_xf"], self.params["R_xo"], self.params["R_xz"], 
        self.params["b_i"], self.params["b_f"], self.params["b_o"], self.params["b_z"]
      )
    
    x = jnp.dot(H, self.params["W_ho"]) + self.params["b_y"]

    x = group_norm(x, num_groups=4)
    x = x[:, :, jnp.newaxis, :]
    x = (jnp.einsum('bijk, kl -> bijl', x, self.params["W_up_1"].T) + 
          self.params["b_up_1"]) * jax.nn.gelu ( (jnp.einsum('bijk, kl -> bijl', x, 
          self.params["W_up_2"].T) + 
          self.params["b_up_2"])
        )

    x = jnp.squeeze(jnp.einsum('bijk, kl -> bijl', x, 
                      self.params["W_dp"]),
                      axis=2
                    ) + self.params["b_dp"]
    x += self.input

    return x

class mLSTM :
    
    def __init__(self, input, 
                       projection_factor=2, 
                       kernel_size=4, 
                       key = random.PRNGKey(100)
                  ) :

        # Input Dimension: (Batch Size, Sequence Length, Embedding dimension)

        assert (input.ndim >= 2)

        if input.ndim == 2:
           input = input[jnp.newaxis, :, :]
    
        keys = {f"key{i+1}": k for i, k in enumerate(random.split(key, num=20))}

        self.input = input
        self.hidden_size = int(input.shape[-1] * projection_factor)
        self.states = {
            "C": jnp.ones((input.shape[0], input.shape[1], self.hidden_size)),
            "N": jnp.ones((input.shape[1], self.hidden_size))
        }

        self.params = {
           "W_i": random.normal(keys["key1"], (self.hidden_size, self.hidden_size)),
           "W_o": random.normal(keys["key2"], (self.hidden_size, self.hidden_size)),
           "W_f": random.normal(keys["key3"], (self.hidden_size, self.hidden_size)),
           "W_q": random.normal(keys["key4"], (self.hidden_size, self.hidden_size)),
           "W_k": random.normal(keys["key5"], (self.hidden_size, self.hidden_size)),
           "W_v": random.normal(keys["key6"], (self.hidden_size, self.hidden_size)),
           "b_i": random.normal(keys["key7"], (1, self.hidden_size)),
           "b_o": random.normal(keys["key8"], (1, self.hidden_size)),
           "b_f": random.normal(keys["key9"], (1, self.hidden_size)),
           "b_q": random.normal(keys["key10"], (1, self.hidden_size)),
           "b_k": random.normal(keys["key11"], (1, self.hidden_size)),
           "b_v": random.normal(keys["key12"], (1, self.hidden_size)),
           "W_up_1": random.normal(keys["key13"], (input.shape[-1], 
                                                   int(input.shape[-1] * projection_factor)
                                                  )),
           "W_up_2": random.normal(keys["key14"], (input.shape[-1], 
                                                   int(input.shape[-1] * projection_factor)
                                                  )),
           "b_up_1": random.normal(keys["key15"], (1, 
                                                   int(input.shape[-1] * projection_factor)
                                                  )),
           "b_up_2": random.normal(keys["key16"], (1, 
                                                   int(input.shape[-1] * projection_factor)
                                                  )),
           "W_dp": random.normal(keys["key17"], (int(input.shape[-1]*projection_factor), 
                                                 int(input.shape[-1]*(projection_factor/2))
                                                )),
           "b_dp": random.normal(keys["key18"], (1, 
                                                 int(input.shape[-1] * (projection_factor/2))
                                                )),
           "kernel": random.normal(keys["key19"], (self.hidden_size, 
                                                   self.hidden_size, 
                                                   kernel_size
                                                  )),
           "skip": random.normal(keys["key20"], (1))
        }

    def cell (self, x, x_conv, 
                    C, N, 
                    W_i, W_f, W_o, W_q, W_k, W_v, 
                    b_i, b_f, b_o, b_k, b_v, b_q
                  ) :
        
        Q = jnp.einsum('bse, eo -> bso', x_conv, W_q) + b_q
        K = ((jnp.einsum('bse, eo -> bso', x_conv, W_k)) / jnp.sqrt(self.hidden_size)) + b_k
        V = jnp.einsum('bse, eo -> bso', x, W_v) + b_v
        I = jnp.exp(jnp.dot(x, W_i) + b_i)
        F = jnp.exp(jnp.dot(x, W_f) + b_f)
        O = jax.nn.sigmoid(jnp.einsum('bse, eo -> bso', x, W_o) + b_o)
        C = F * C + jnp.einsum('bxy, bac -> bxc', (V @ jnp.transpose(K, (0, 2, 1))), I)
        N = F * N + I * K
        H = O * ((C * Q) / jnp.max(jnp.transpose(N, (0, 2, 1)) @ Q, 1))

        return H, (C, N)
    
    def forward (self) :
       
       x = layer_norm(self.input)
       x1 = jnp.einsum('bse, eu -> bsu', x, self.params["W_up_1"]) + self.params["b_up_1"]
       x2 = jnp.einsum('bse, eu -> bsu', x, self.params["W_up_2"]) + self.params["b_up_2"]
       x1_conv = jax.nn.swish(causal_conv1d(x1, self.params["kernel"]))

       H, _ = self.cell(
          x1, x1_conv, 
          self.states["C"], self.states["N"],
          self.params["W_i"], self.params["W_f"], self.params["W_o"], self.params["W_q"],
          self.params["W_k"], self.params["W_v"], self.params["b_i"], self.params["b_f"],
          self.params["b_o"], self.params["b_k"], self.params["b_v"], self.params["b_q"] 
       )
       x = group_norm(H)
       x += (self.params["skip"] * x1_conv)
       x *= jax.nn.swish(x2)

       x = jnp.einsum('bse, eu -> bsu', x, self.params["W_dp"]) + self.params["b_dp"]
       x += self.input

       return x


x = random.normal(random.PRNGKey(100), (1, 50, 128))
xLSTM = mLSTM(sLSTM(x, 256).forward()) # There could be multiple such blocks, in any order
y = xLSTM.forward()
```
*The paper mentions 'heads' between each cells, which simply means splitting of the inputs and processing
them in parallel, before concatenating them again. I have left it out, since I did not perform parallel processing, although it is trivial to add here. A detailed implementation is in their official [repository](https://github.com/NX-AI/xlstm/tree/main/xlstm/blocks/slstm).*