---
title: "Sparse Auto Encoders"
date: 2024-12-10
draft: false
references:
  - title: "Sparse Autoencoders find highly interpretable features in LLMs"
    url: "https://arxiv.org/abs/2309.08600"  
  - title: "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    url: "https://transformer-circuits.pub/2023/monosemantic-features"
---

While the extends of LLMs capabilities have been deeply scrutinized and well tested, only recently has there been an effort in understanding the internals of LLMs. One of the most prominent and popular approach was the use of _sparse_ auto-encoders, which help us map the internal _polysemantic_ features of llms to more human-understandable, _monosemantic_ features.

<br>

## Equation

\begin{flalign}
&\hat{x} = M^T \cdot ReLU(Mx + b)\newline
&\mathcal{L} = \overbrace{||x - \hat{x}||^2_{2}}^{L_{1}} + \overbrace{\alpha||c||_{1}}^{L_2}
\end{flalign}

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>x</i></td>
    <td style="vertical-align: middle;">: Input vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">x&#x0302;</td>
    <td style="vertical-align: middle;">: Reconstructed input</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>M</i></td>
    <td style="vertical-align: middle;">: Feature Dictionary (parameters)</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>b</i></td>
    <td style="vertical-align: middle;">: Bias</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>L</i><sub>1</sub></td>
    <td style="vertical-align: middle;">: Reconstruction Loss</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>L</i><sub>2</sub></td>
    <td style="vertical-align: middle;">: Sparsity Loss</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">𝓛</td>
    <td style="vertical-align: middle;">: Objective Function</td>
  </tr>
</table>

<br>

## Explanation

- Neurons in MLP weights of a large language model can respond to multiple features, a phenomenon termed as _polysemanticity_. This is acheived by a technique called _superposition_, and overall makes the interpretation process tedious and infeasible. The features are not human understandable. (For a more in-depth explanation on _Polysemanticity_ and _Superposition_, read [these](https://transformer-circuits.pub/) papers).

- Here, our sparse autoencoder was used to learn a **feature dictionary** by reconstructing the original input from sparse internal representations. These learned features correspond to internal feature maps of the LLMs. The second term in the loss function—the **sparsity penalty**—encourages the activations to be sparse, which promotes _monosemanticity_: each feature tends to represent a single, interpretable concept. This aids interpretability, as dense representations often exhibit _superposition_, where individual neurons encode multiple unrelated concepts, making them _polysemantic_.

<br>

## Code

```python
from jax import random
from typing import Tuple
import jax, jax.numpy as jnp

Key = jax.random.PRNGKey
Array = jnp.ndarray

def init(key: Key, d: int, h: int) -> Tuple[Array, Array]:
    k1, _ = random.split(key)
    M = random.normal(k1, (h, d)) * jnp.sqrt(2 / d)
    b = jnp.zeros(h)
    return M, b

def ae(M: Array, b: Array, X: Array) -> Tuple[Array, Array]:
    C = jax.nn.relu(X @ M.T + b)
    Xh = C @ M
    return Xh, C

def loss(M: Array, b: Array, X: Array, α: float) -> Array:
    Xh, C = ae(M, b, X)
    return jnp.mean(jnp.sum((X - Xh) ** 2, -1) + α * jnp.sum(jnp.abs(C), -1))

key = random.PRNGKey(0)
d, h, n = 64, 32, 16
M, b = init(key, d, h)
X = random.normal(key, (n, d))
α = 1e-3
L = loss(M, b, X, α)
```