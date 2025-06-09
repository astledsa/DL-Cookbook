---
title: "Contractive De-noising Auto Encoders"
date: 2024-12-10
draft: false
references:
  - title: "Contractive Denoising AutoEncoder"
    url: "https://arxiv.org/abs/1305.4076"  
---

Among various variants that were innovated in order to improve upon auto-encoders, we shall focus on two: _denoising_ and _contractive_ autoencoders, their advantages and how in this paper, the authors seek to combine them into one architecture: the contractive denoising auto encoder.

<br>

## Equation

\begin{flalign}
&L_{DAE} = \frac{1}{N}\sum_{x\in D}((x - x_{rec})^2 + \lambda||J_{\tilde{h}}(\tilde{x})||^2_{F})
\end{flalign}

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>x</i></td>
    <td style="vertical-align: middle;">: Input vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">&#x7e;x</td>
    <td style="vertical-align: middle;">: Corrupted Input vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">&#x7e;h</td>
    <td style="vertical-align: middle;">: Corrupted hidden vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">x<sub>rec</sub></td>
    <td style="vertical-align: middle;">: Output of decoder</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">𝜆</td>
    <td style="vertical-align: middle;">: Scalar</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>J<sub>f</sub></i></td>
    <td style="vertical-align: middle;">: Jacobian matrix from function <i>f</i></td>
  </tr>
</table>

<br>

## Explanation

- **Denoising** VAE adds a simple step before each forward pass: they **corrupt** the input vector, and force the model to learn the _true_ value from the corrupted output. One way to corrupt a vector is to add slight gaussian noise before being processed (standard gaussian in most cases). The reason for this, especially in cases where we need the model to learn the _generator_ distribution of the input, is that the model optimizes for _slightly_ different inputs (corrupted ones), and hence gets better at approximation and more robust to our input distribution.

- **Contractive** AEs tend to take a different approach in order to make the models robust to slight variations and tweaks in inputs, by _penalizing the frobenuis norm of the Jacobian_ of the inputs with respect to the model. A few terms to expand on from the previous sentence: the jacobian of a variable is basically it's _derivative_ with respect to a certain function, but since it's a matrix/vector, it'll have the same higher dimensions. The Frobenuis norm of any matrix is simply a way to measure the "size" of the matrix (similar to the norm of a vector). By penalizing the frobenuis norm of the jacobian, we will essentially restrict _how much the parameters can change_ due to slight variations in the inputs, making them more "resilient" to small changes.

- Bringing both of the above architectural choices into one, we get the _contractive denoising auto encoder_, wherein we both corrupt the input and penilize the frobenuis norm of the _corrupted_ output, hence making the overall model more robust.

<br>

## Code

```python
import jax, jax.numpy as jnp
from jax import random, vmap, jacrev
from typing import Sequence, List, Tuple

Key = jax.random.PRNGKey
Params = List[Tuple[jnp.ndarray, jnp.ndarray]]

def init_mlp(sizes: Sequence[int], key: Key) -> Params:
  ks = random.split(key, len(sizes) - 1)
  return [(random.normal(k, (a, b)) * jnp.sqrt(2 / a), jnp.zeros(b))
    for k, a, b in zip(ks, sizes[:-1], sizes[1:])]

def mlp(p: Params, x: jnp.ndarray) -> jnp.ndarray:
  for w, b in p[:-1]:
    x = jnp.tanh(x @ w + b)
  w, b = p[-1]
  return x @ w + b

def encode(p: Params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  h = mlp(p, x)
  return jnp.split(h, 2, -1)

def decode(p: Params, z: jnp.ndarray) -> jnp.ndarray:
  return mlp(p, z)

def loss(
    p_e: Params, p_d: Params, 
    x: jnp.ndarray, k: Key,
    σ: float, λ: float
  ) -> jnp.ndarray:
  k1, k2 = random.split(k)
  x̃ = x + σ * random.normal(k1, x.shape)
  μ, logσ2 = encode(p_e, x̃)
  z = μ + jnp.exp(0.5 * logσ2) * random.normal(k2, μ.shape)
  x̂ = decode(p_d, z)
  mse = jnp.mean((x - x̂) ** 2)
  f = lambda y: encode(p_e, y)[0]
  J = vmap(jacrev(f))(x̃)
  c = jnp.mean(jnp.sum(J ** 2, axis=(1, 2)))
  return mse + λ * c

key = random.PRNGKey(0)
x = random.normal(key, (32, 64))
σ = 0.1
λ = 1e-3

enc_sizes = [64, 128, 32 * 2]
dec_sizes = [32, 128, 64]
k1, k2, k3 = random.split(key, 3)
p_enc = init_mlp(enc_sizes, k1)
p_dec = init_mlp(dec_sizes, k2)

L = loss(p_enc, p_dec, x, k3, σ, λ)
```

*<strong>NOTE</strong>: vmap function of jax is the simple map function, but for vectors (hence vectorized map, vmap). For more information, visit the [docs](https://docs.jax.dev/en/latest/automatic-vectorization.html)*
