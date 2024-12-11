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

- The authors also propose a custom backpropagation algorithm, which speeds up the calculation of the intermediary variables by reusing the stored values, which results in faster calculation (through analytic solutions) and thus leads to a linear scale in complexity.

<br>

## Jax Code

```python
import jax
import jax.lax as lax
from jax import random
import jax.numpy as jnp
from jax import custom_vjp
from functools import partial
from concurrent.futures import ThreadPoolExecutor

key = random.PRNGKey(100)
key1, key2, key3, key4, key5, key6, key7, key8 = random.split(key, num=8)

# Important numbers
batch_size=32
seq_len = 100
hidden_size = 256
embedding_dim = 128

# Initilalising matrices
x = random.normal(key1, (batch_size, seq_len, embedding_dim))
W_q = random.normal(key2, (embedding_dim, hidden_size))
W_k = random.normal(key3, (embedding_dim, hidden_size))
W_v = random.normal(key4, (embedding_dim, hidden_size))
W_O = random.normal(key4, (hidden_size, embedding_dim))
b_q = random.normal(key5, (hidden_size,))
b_k = random.normal(key6, (hidden_size,))
b_v = random.normal(key7, (hidden_size,))
b_o = random.normal(key8, (embedding_dim,))

def _causal_mask_linear (self, φQ, φK, V) :
  """
  A naive implementation of the causal linear attention.
  This method is slower compared to the other two implementions,
  but since the paper mentions it in this format, I have decided
  to include it as a reference.
  """
  N = V.shape[1]
  _V = jnp.zeros_like(V)
  S = jnp.zeros((self.hidden_size, self.hidden_size))
  for i in range(N) :
    S = S + jnp.dot(φK[:, i, :].T, V[:, i, :])
    _V = _V.at[:, i, :].set(jnp.dot(φQ[:, i, :], S))
  return _V

def _parallel_causal_mask_linear (φQ, φK, V) :
  """
  Since the paper only mentions the algorithm, I implemented a
  faster version of the causal mask attention, through utilising
  the associative scan and multi-threading functions.
  """
  (batch_size, N, hidden_size) = V.shape
  _V = jnp.zeros_like(V)
  S = jnp.zeros((hidden_size, 1, hidden_size))
  _S = lax.associative_scan(jnp.add, 
                              jnp.concatenate(
                                  (S, jnp.einsum("bij,jik->bik", φK.transpose((2, 1, 0)), V)
                                  ), axis=1), 
                              axis=1
                            )
    
  def update_slice(i):
    global _V
    _V = _V.at[:, i, :].set(jnp.dot(φQ[:, i, :], _S[:, i, :]))
  with ThreadPoolExecutor(max_workers=N) as executor:
    executor.map(update_slice, range(N))
  return _V

def _fullLinearAttention (φQ, φK, V):
  """
  The complete linear attention, without the causal mask. This
  implementation is faster because of the internal optimizations
  done by XLA compiler. Note: this is mathematically different
  than the causal mask attention above.
  """
  φKV = jnp.einsum("nsd,nsm->nmd", φK, V)
  Z = 1/(jnp.einsum("nld,nd->nl", φQ, φK.sum(axis=1)))
  V = jnp.einsum("nld,nmd,nl->nlm", φQ, φKV, Z)
  return V

@partial(custom_vjp)
def linearAttention (φQ, φK, V) :
  """
  The authors also provide with a custom backpropagation
  method, which brings the complexity to linear scale, as 
  opposed to a naive implementation of the autograd algorithm.
  The custom backprop is only in case of causal masking, and
  not for the full linear attention mechanism (though I think
  the derivation would not be that different)
  A = _fullLinearAttention(φQ, φK, V)
  """
  A = _parallel_causal_mask_linear(φQ, φK, V)
  intermediates = (φQ, φK, V)
  return A, intermediates

def _attentionWeights_forward(φQ, φK, V):
  return linearAttention(φQ, φK, V)

def _attentionWeights_backward(saved_values, gA):
  """
  The custom backpropagation saves memory through utilising
  intermediate results and hence results in a linear scaling
  with the sequence length. I Although not mentioned in the
  paper itself, I have used the associative scan + multi-threading
  here as well. This method does result in a speed up, but there
  can be more efficient methods that could be explored in the
  official repository.
  """
  φQ, φK, V = saved_values
  N = V.shape[1]
  gφQ = jnp.zeros_like(φQ)
  gφK = jnp.zeros_like(φK)
  gV = jnp.zeros_like(V)

  S = jnp.zeros((hidden_size, 1, hidden_size))
  _S = lax.associative_scan(jnp.add, 
                            jnp.concatenate(
                                (S, jnp.einsum("bij,jik->bik", φK.transpose((2, 1, 0)), V)
                                ), axis=1), 
                            axis=1
                          )
  def update_slice(i):
    global gφQ
    gφQ = gφQ.at[:, i, :].set(jnp.dot(gA[:, i, :], _S[:, i, :].transpose(2, 1, 0)))
  with ThreadPoolExecutor(max_workers=N) as executor:
    executor.map(update_slice, range(N))
  
  S = jnp.zeros((hidden_size, 1, hidden_size))
  _S = lax.associative_scan(jnp.add, 
                            jnp.concatenate(
                                (S, jnp.einsum("bij,jik->bik", φQ.transpose((2, 1, 0)), gA)
                                ), axis=1), 
                            axis=1
                          )
  def update_slice(i):
    global gφK
    global gV
    gV = gV.at[:, i, :].set(jnp.dot(φK[:, i, :], _S[:, i, :].transpose(2, 1, 0)))
    gφK = gφK.at[:, i, :].set(jnp.dot(V[:, i, :], _S[:, i, :]))
  with ThreadPoolExecutor(max_workers=N) as executor:
    executor.map(update_slice, range(N))
  
  return (gφQ, gφK, gV)

linearAttention.defvjp(_attentionWeights_forward, _attentionWeights_backward)

class LinearAttention :

  def __init__(self, hidden_size, embedding_dim, causal=True):
    self.causal = causal
    self.hidden_size = hidden_size
    self.embedding_dim = embedding_dim
    self.weights = {
        "W_q": random.normal(key2, (embedding_dim, hidden_size)),
        "W_k": random.normal(key3, (embedding_dim, hidden_size)),
        "W_v": random.normal(key4, (embedding_dim, hidden_size)),
        "W_O": random.normal(key4, (hidden_size, embedding_dim)),
        "b_q": random.normal(key5, (hidden_size,)),
        "b_k": random.normal(key6, (hidden_size,)),
        "b_v": random.normal(key7, (hidden_size,)),
        "b_o": random.normal(key8, (embedding_dim,))
    }
  
  def forward_pass (self, x) :
    Q = jnp.dot(x, self.weights["W_q"]) + self.weights["b_q"]
    K = jnp.dot(x, self.weights["W_k"]) + self.weights["b_k"]
    V = jnp.dot(x, self.weights["W_v"]) + self.weights["b_v"]
    φQ = jax.nn.elu(Q) + 1
    φK = jax.nn.elu(K) + 1
    
    return linearAttention(φQ, φK, V)

  def forward (self, x) :
    A = self.forward_pass(x)
    y = jnp.dot(A[0], self.weights["W_O"]) + self.weights["b_o"]
    return y

model = LinearAttention (hidden_size, embedding_dim)
y = model.forward(x)
```