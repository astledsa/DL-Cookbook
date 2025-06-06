---
title: "Vector Quantized Variational AutoEncoders"
date: 2025-06-05
draft: false
references:
  - title: "Neural Discrete Representational Learning"
    url: "https://arxiv.org/abs/1711.00937"  
  - title: "Vector Quantization"
    url: "https://ieeexplore.ieee.org/document/1162229"
---

Following the release of Variational AutoEncoders, as is the case for every other model architecture, the AI research community saw an influx of _variants_ of VAEs, each trying to solve a problem observed in the original paper. This particular variant is one of the most successful ones, while also utilising a prominent concept from deep learning (vector embeddings) quite intimately.
Note: as in the case of VAE, the below given equation is a **loss function**, and not just the forward pass, since in these cases, the loss equation has more significance then the forward pass (where the implementation of a forward pass is left as a choice).

<br>

## Equation

\begin{flalign}
&\mathcal{L} = \overbrace{log(p(x|z_{q}(x)))}^{L_{1}} + \overbrace{||sg[z_e(x)] - e||^2_{2}}^{L_{2}} + \overbrace{\beta||z_{e}(x) - sg[e]||^2_{2}}^{L_{3}}
\end{flalign}

<br>

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>L</strong><sub>1</sub></td>
    <td style="vertical-align: middle;">: Reconstruction Loss</td>
  </tr>
   <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>L</strong><sub>2</sub></td>
    <td style="vertical-align: middle;">: Embedding Loss</td>
  </tr>
   <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>L</strong><sub>3</sub></td>
    <td style="vertical-align: middle;">: Commitment Loss</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>z</i><sub>e</sub>(x)</td>
    <td style="vertical-align: middle;">: Encoder output</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>z</i><sub>q</sub>(x)</td>
    <td style="vertical-align: middle;">: Quantized encoder output</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;">p(x|z<sub>q</sub>(x))</td>
    <td style="vertical-align: middle;">: Identity loss</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>sg[ ]</i></td>
    <td style="vertical-align: middle;">: Stop Gradient operator</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>β</i></td>
    <td style="vertical-align: middle;">: Scalar threshold</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>e</i></td>
    <td style="vertical-align: middle;">: Embedding vectors</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>x</i></td>
    <td style="vertical-align: middle;">: Input vectors</td>
  </tr>
</table>

<br>

## Explanation

- As we remember from the lesson on VAEs, auto encoding generative models seek to minimaze a loss function known as the Evidence Lower BOund, or ELBO (the above given equation in our case). The theory of how we get to this particular loss function is very similar (or rather the same) as that of VAEs, hence we should only focus on the _differences_.

- The first distinction is a change in assumptions: the authors assume that the distribution of the latent variable _z_, is **discrete**, as opposed to continous in the original paper. This led to the replacement of the reparametrization trick with a [**vector embedding** table](https://en.wikipedia.org/wiki/Vector_quantization), which was a simple dictionary that mapped encoding input vectors to embedding vectors, further to be processed by the decoder. This dictionary is also called a _codebook_ (or sometimes a _cookbook_).

- The discretization, as argued by the authors, leads to a comparatively better representation of the generator distribution, as the various modalities of data we need to generate are discrete by nature (like images, language or speech). Another failure mode that was avoided was the **posterior collapse**, a phenomem brought forth due to the usage of a powerful autoregressive decoder, which learns to _skip the contributions/influence of the latent variable_ and directly model our inputs, _x_.

- The first term of the loss function is called the **reconstruction loss**, which optimizes the encoder and the decoder to better model our input distribution (in practice, I have implemented a simple MSE loss).

- The second term of the loss function is simply the **embeddings loss**, which optimizes the embedding space using the VQ algorithm with an l<sub>2</sub> loss. The purpose of this term is to minimize the distance between the encoder's outputs and our embedding vectors _e_. The stop gradient operator basically stops the flow of gradient from this loss function into the encoder, and hence the encoder outputs remain a constant in that term (this is done since the second term is to mainly optimize the dictionary).

- Since the embedding space is _dimensionless_, the encoder might never settle into an embedding space. To counter this, the third loss term is utilised, called the **commitment loss**. This loss term minimizes the distance between the encoders and a _constant_ embedding space (which is kept constant by ceasing the flow of gradient to the embedding vectors, through the stop gradient operator). This makes sure that the encoder learns to map close to an embedding space while training.

<br>

## Code

```python
import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Dict, Sequence, Tuple, List, Callable

Array = jax.Array
LinearParams = Dict[str, Array]
MLPParams = List[LinearParams]
VQVAEParams = Dict[str, Any]

# The glorot initialization (https://shorturl.at/aSUj6)
def glorot_init(key: Array, fan_in: int, fan_out: int) -> Array:
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)


def init_linear(key: Array, in_dim: int, out_dim: int) -> LinearParams:
    k1, _ = random.split(key)
    w = glorot_init(k1, in_dim, out_dim)
    b = jnp.zeros((out_dim,))
    return {"W": w, "b": b}

def linear(params: LinearParams, x: Array) -> Array:
    return x @ params["W"] + params["b"]

def init_mlp(key: Array, sizes: Sequence[int]) -> MLPParams:
    keys = random.split(key, len(sizes) - 1)
    return [init_linear(k, sizes[i], sizes[i + 1]) for i, k in enumerate(keys)]

def mlp_forward(
    params: MLPParams,
    x: Array,
    final_activation: Callable[[Array], Array] | None = None,
) -> Array:
    *hidden, last = params
    h = x
    for p in hidden:
        h = jax.nn.relu(linear(p, h))
    h = linear(last, h)
    if final_activation is not None:
        h = final_activation(h)
    return h


def quantize(z_e: Array, codebook: Array) -> Tuple[Array, Array]:
    dists = jnp.sum((z_e[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    indices = jnp.argmin(dists, axis=1)
    z_q = codebook[indices]
    return z_q, indices


def vq_vae_loss(
    x: Array,
    x_recon: Array,
    z_e: Array,
    z_q: Array,
    beta: float,
) -> Tuple[Array, Dict[str, Array]]:
    recon = jnp.mean((x - x_recon) ** 2)
    codebook = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
    commit = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
    total = recon + codebook + beta * commit
    return total, {
        "recon_loss": recon,
        "codebook_loss": codebook,
        "commitment_loss": commit,
    }


def init_vqvae_params(
    key: Array,
    input_dim: int = 784,
    encoder_sizes: Sequence[int] = (128, 64),
    decoder_sizes: Sequence[int] = (128, 784),
    codebook_size: int = 512,
    embed_dim: int = 64,
) -> VQVAEParams:
    k_enc, k_dec, k_cb = random.split(key, 3)
    enc = init_mlp(k_enc, [input_dim, *encoder_sizes, embed_dim])
    dec = init_mlp(k_dec, [embed_dim, *decoder_sizes])
    codebook = random.normal(k_cb, (codebook_size, embed_dim))
    return {"encoder": enc, "decoder": dec, "codebook": codebook}


def vqvae_forward(
    params: VQVAEParams,
    x: Array,
    beta: float = 0.25,
) -> Tuple[Array, Dict[str, Array], Array]:
    z_e = mlp_forward(params["encoder"], x)
    z_q, _ = quantize(z_e, params["codebook"])
    x_recon = mlp_forward(params["decoder"], z_q)
    loss, metrics = vq_vae_loss(x, x_recon, z_e, z_q, beta)
    return loss, metrics, x_recon


key = random.PRNGKey(0)
batch_size = 32
input_dim = 784
params = init_vqvae_params(key, input_dim=input_dim)
x_key, _ = random.split(key)
x_batch = random.normal(x_key, (batch_size, input_dim))
loss, metrics, _ = vqvae_forward(params, x_batch)
```