---
title: "Sparse Transformers"
date: 2024-11-26
draft: false
references:
  - title: "Generating long sequences with sparse transformers"
    url: "https://arxiv.org/abs/1904.10509"  
  - title: "Big Bird: Transformers for long sequences"
    url: "https://arxiv.org/abs/2007.14062"
---

Another variant of the attention mechanism, also created to mitigate the quadratic complexity, is the sparse (or factorized, as called in the paper) attention mechanism, which essentially forces the operation to "attend" to only certain parts of the sequences. This side steps the unnecessary computation of the entire provided sequence (as is done in the vanilla transformers).

<br>

## Equation

![Sparse Attention Diagram](/images/sparse2.png)

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
    <td style="padding-right: 20px; vertical-align: middle;"><i>S</i><sup> i</sup></td>
    <td style="vertical-align: middle;">: Masks</td>
  </tr>
</table>

<br>

## Explanation

- The main idea behind this architecture was to replace the dense attention matrix with a sparser one to increase speed, while also preserving accuracy. The proposed _masks_ contain information as to which indices must be included in the operation and which must be ignored (theoretically, it's a binary matrix, but in practice we use large negative value). The **strided** attention pattern is suited for 2D images, as it captures the local pixel-level dependanceis (S<sup>1</sup> and S<sup>2</sup> in the above equations), while the other two (S<sup>3</sup> and S<sup>4</sup>) are more suited for sequence processing or **fixed** attention pattern.

- In images or aligned data, the strided attention pattern allows a transformer to efficiently look at nearby positions (previous *l* positions) and also periodically check distant, strategically-selected positions (previous *lth* positions). This approach reduces computational complexity while maintaining contextual understanding across different spatial scales. (S<sup>1</sup> and S<sup>2</sup>)

- In text sequences, unlike images, standard attention patterns might not capture meaningful spatial relationships. To address this, the authors propose fixed attention patterns where specific "anchor" cells selectively capture and propagate information across future cells, enabling more structured and meaningful information flow in sparse transformers. (S<sup>3</sup> and S<sup>4</sup>)

- With four potential masks, there are various ways in which these can be used in a sparse transformer architecture. The three ways provided by the authors are : i) Using each mask **interleaved** between different residual blocks (according to whether you're dealing with images or text), ii) **merging** the masks before applying them through a hadamard product operation or iii) using **multi-head** attention pattern, where different masks are used for different heads and calculated in parallel.

<br>

## Jax code

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

def scaled_dot_product_attention(q, k, v, mask):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    attention_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, v)
    return output, attention_weights

# Strided masks for images
def Strided(seq_len, stride):
    mask = jnp.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
      mask = mask.at[i, max(0, i - stride):i + 1].set(True)
    return mask.astype(float)

# Fixed masks for sequences
def Fixed(seq_len, block_size):
    mask = jnp.zeros((seq_len, seq_len), dtype=bool)
    for i in range(0, seq_len, block_size):
        mask = mask.at[i:i + block_size, i:i + block_size].set(True)
    return mask.astype(float)

def SparseAttention (W_q, W_k, W_v, x, mask="strided", stride=4, block=4):
  
  seq_len = x.shape[1]
  Q = jnp.dot(x, W_q) + b_q
  K = jnp.dot(x, W_k) + b_k
  V = jnp.dot(x, W_v) + b_v

  # Initiate the mask
  if mask == "strided":
    attentionMask = Strided(seq_len, stride)
  else:
    attentionMask = Fixed(seq_len, block)
  
  # Reshape the mask and calculate softmax
  attentionMask = jnp.broadcast_to(attentionMask, (batch_size, seq_len, seq_len))
  return scaled_dot_product_attention(Q, K, V, attentionMask)

A, _ = SparseAttention(W_q, W_k, W_v, x)
y = jnp.dot(A, W_O) + b_o
```