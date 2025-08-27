---
title: "Negative Weights Transformers"
date: 2024-12-10
draft: false
references:
  - title: "More Expressive Attention with Negative Weights"
    url: "https://arxiv.org/html/2411.07176v1#:~:text=Negative%20weights%20reduce%20effective%20information,diffusion%20models%20for%20image%20generation."  
---

Expressivity in a model means the total _range_ of features it can represent, that are present in a dataset. This is indeed a very important property that any machine learning model must possess in order to accurately approximent the input's distribution. This particular variant of transformer set out to enhance the _expressivity_ of the self-attention mechanism by allowing for _negative_ values.

<br>

## Equation

\begin{align}
Q=W_qx+b_q\newline
K=W_kx+b_k\newline
V=W_vx+b_v\newline
y=W_oA+b_o\newline
\newline
where,\newline
\end{align}

\begin{align}
A=\frac{SignExp(QK^T)}{\sum SignExp(QK^T)}\newline
\newline
SignExp(p_{i,j}) = s_{i,j} \cdot exp(s_{i,j} \cdot p_{i,j} - \hat{m}_{i})\newline
\end{align}

\begin{align}
s_{i,j} = sign(p_{i,j})\, \quad \hat{m}_i = max(|p_i|)
\end{align}

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
    <td style="vertical-align: middle;">: Query matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>K</strong><i></i></td>
    <td style="vertical-align: middle;">: Key matrices</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>A</strong><i></i></td>
    <td style="vertical-align: middle;">: Cog Attention matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>s</strong></td>
    <td style="vertical-align: middle;">: Signed Matrix</td>
  </tr>

</table>

<br>

## Explanation

- As mentioned above, the _expressivity_ of a model is perhaps the most important aspect of an architecture. In this variant of transformers, the authors put forth a simple yet under-explored question: _"Why not let the attention values be negative ?"_ And explore the subsequent reasons and their potential solutions. 

- The softmax attention mechanism essentially _normalizes_ the attention matrix, since the **QK** matrix includes negative values already. The reason for normalization is both training stability and fact that the total of the attention weights should be a constant, 1. If we were to forgo the softmax operation, we loose both of these guarantees. Since the attention matrix can only have positive values, the _operations_ that are needed to be performed on the attention weights (like deletion or copying) must be handled by the **OV** matrix, since non-negativity is a _constraint_ (The operations are performed as needed by the context). 

- The in order to let the attention matrix have negative weights, any proposed operation must provide solutions to the problems: training stability and normalization. In order to satisfy all _three_ conditions (negative weights + stability + normalization), the authors propose **cog attention** mechanism (the `SignExp()` above). The exponential function of softxmax is retained, as it proves better for numerical stability (with the subtraction of the maximum value for stability). The rest of the formula is straightforward: the inclusion of `sign()` retains the sign of the value, and in order to avoid `NaN` errors, the authors sum the absolute values of the outputs of `SignExp()` as the denominator. 

<br>

## Code

```python
from jax import random
import jax, jax.numpy as jnp
from typing import Dict, Tuple

Key = jax.random.PRNGKey
Array = jnp.ndarray
Params = Dict[str, Array]

def init_tr(key: Key, d: int) -> Params:
    k = random.split(key, 8)
    return dict(
        Wq=random.normal(k[0], (d, d)) / jnp.sqrt(d),
        Wk=random.normal(k[1], (d, d)) / jnp.sqrt(d),
        Wv=random.normal(k[2], (d, d)) / jnp.sqrt(d),
        Wo=random.normal(k[3], (d, d)) / jnp.sqrt(d),
        bq=jnp.zeros(d),
        bk=jnp.zeros(d),
        bv=jnp.zeros(d),
        bo=jnp.zeros(d),
    )

def cog_attn(q: Array, k: Array, v: Array, mask: Array) -> Array:
    p = jnp.einsum("bld,bmd->blm", q, k)
    abs_p = jnp.where(mask, jnp.abs(p), -jnp.inf)
    a = jnp.sign(p) * jax.nn.softmax(abs_p, -1)
    return jnp.einsum("blm,bmd->bld", a, v)

def tr_forward(p: Params, x: Array, mask: Array) -> Array:
    q = jnp.einsum("bld,dd->bld", x, p["Wq"]) + p["bq"]
    k = jnp.einsum("bld,dd->bld", x, p["Wk"]) + p["bk"]
    v = jnp.einsum("bld,dd->bld", x, p["Wv"]) + p["bv"]
    h = cog_attn(q, k, v, mask)
    return jnp.einsum("bld,dd->bld", h, p["Wo"]) + p["bo"]

key = random.PRNGKey(0)
b, L, d = 2, 4, 8
params = init_tr(key, d)
x = random.normal(key, (b, L, d))
mask = jnp.broadcast_to(jnp.tril(jnp.ones((L, L))).astype(bool), (b, L, L))
y = tr_forward(params, x, mask)
```

_**NOTE:** The output is a three dimensional matrix due to operations being performed batch-wise._