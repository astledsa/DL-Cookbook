---
title: "Variational Auto Encoders"
date: 2024-12-10
draft: false
references:
  - title: "Auto-Encoding Variational Bayes"
    url: "https://arxiv.org/abs/1312.6114"  
---

Pivoting to a slightly different yet extremely interesting branch of model architectures, we arrive at the auto-encoders. While the theoretical underpinnings might prove quite cumbersome, the implementation in code seems just as easy to follow (this being a recurring theme in computer science). The auto encoders can be viewed as a class of _generative_ models, which ease the approximations of intractable input spaces/distributions through clever use of probability and statistics.

<br>

## Equation

\begin{flalign}
&\mathcal{L}(\theta, \phi;X) = \frac{1}{2} \sum_{j=1}^{J} (1 + log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)  + \frac{1}{L} \sum_{l=1}^{L} log(p_\theta(X | Z^{(l)}))\newline
&\newline
&where,\newline
&q_\phi(z|X) = \mathcal{N}(z;\mu, \sigma^2I)
\end{flalign}

<br>

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>X</strong></td>
    <td style="vertical-align: middle;">: Input Matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>Z</strong></td>
    <td style="vertical-align: middle;">: Latent variable</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>𝓛</i></td>
    <td style="vertical-align: middle;">: Evidence Lower Bound (ELBO)</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>𝒩</i></td>
    <td style="vertical-align: middle;">: Standard Gaussian</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>θ</i></td>
    <td style="vertical-align: middle;">: Encoder Parameters</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>φ</i></td>
    <td style="vertical-align: middle;">: Decoder Parameters</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>μ</i></td>
    <td style="vertical-align: middle;">: Standard Mean</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>σ</i></td>
    <td style="vertical-align: middle;">: Standard Deviation</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>q<sub>φ</sub>(z|x)</i></td>
    <td style="vertical-align: middle;">: Probabilistic Encoder</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>p<sub>θ</sub>(x|z)</i></td>
    <td style="vertical-align: middle;">: Probabilistic Decoder</td>
  </tr>
</table>

<br>

## Explanation

- Our entire journey starts with the purpose of trying to find out the probability distribution that generated our Input dataset (say _**X**_), or from which it was randomly sampled by. This, in reality, is a problem. For one, the distribution itself can be extremely hard to approximate, or in a lot of cases, even impossible or intractable. Another reason is our dataset can be influenced by hidden or _**latent**_ variables, (_**z**_ in our case), which we must consider in our model to accurately approximate the _generator_ distribution of our input.

- The probability of *x* occuring, given that *z* was observed, is denoted by <i>p(x|z)</i>. If the distribution is generated using parameters θ, then we say the distribution is _parametrized_ by θ, denoted as p<sub>θ</sub>(x|z). Here, we are interested in (at least partly) this: p<sub>θ</sub>(z|x) or the probability of *z*, given *x* was observed. We call it the **true posterior**. Bayes theorem states: $$p(z|x) = \frac{p(z)p(x|z)}{p(x)}$$ RHS of the above equation is intractable in real life: not every dataset can be accurately modelled by our equations and methods. Here, we turn to a _recognition model_ (a neural network in practice) to approximate the true posterior, denoted by <i>q<sub>φ</sub>(z|x)</i> (also called the _encoder_). 

- The details of how and why we get to the above given ELBO or evidence lower bound, and why do we focus on minimizing _that_ particular equation involved arduous amount of derivation, hence I would simply provide a sketch here, _without_ any equations. 
  
  - We start with the **marginal likelihood** or *p(x)*, the total probability of X occuring, which includes the latent variables as well. Through derivation, we conclude that this value is the addition of the (Kullback-Leibler divergence between <i>q<sub>φ</sub>(z|x)</i> and p<sub>θ</sub>(z|x)) and our evidence lower bound.
  - The KL divergence is akways non-negative, hence the other term must be the _variational_ lower bound: something we can work towards minimizing. Through another few steps of derivation, we arrive at the lower bound being the addition of the KL divergence and our estimation of the _decoder_. 
  - The **reparametrization** trick tells us that it is often possible to represent the latent variable's distribution with another deterministic equation, namely the standard gaussian. This trick makes our final equation much easier to compute, and makes our final loss function tractable. The choice of standard gaussian is arbitary, in classification tasks, it can also be bernoulli.

- The final loss function (ELBO) thus consists of two terms: the KL divergence of our estimation of the true posterior (through an MLP) and the standard guassian, and the decoder outputs. This makes our loss function well behaved, and has been shown to work well empirically. While the theory behind the VAE may be non-trivial (I have not shown any derivations), the implimentation is straightforward. For detailed derivations and formulae (or how we got to the above given one), refer to the resources given, especially the appendices of the paper. (NOTE: the AEVB loss mentioned below is simply the AutoEncoding Variational Bayes loss, which is simply the negative ELBO)

<br>

## Code

```python
import jax
import jax.numpy as jnp
from jax import random

def probabilistic_encoder(params, x):
  hidden = jnp.tanh(jnp.dot(x, params['W1']) + params['b1'])
  μ = jnp.dot(hidden, params['W_μ']) + params['b_μ']
  log_σ = jnp.dot(hidden, params['W_σ']) + params['b_σ']
  return μ, log_σ

def decoder(params, z):
  hidden = jnp.tanh(jnp.dot(z, params['W1']) + params['b1'])
  reconstructed_mean = jnp.dot(hidden, params['W_out']) + params['b_out']
  return reconstructed_mean

def reparameterize(rng_key, μ, log_σ):
  epsilon = random.normal(rng_key, μ.shape)
  σ = jnp.exp(log_σ)
  z = μ + σ * epsilon
  return z

def gaussian_log_likelihood(x, reconstructed_mean, log_variance=0.0):
  variance = jnp.exp(log_variance)
  return -0.5 * jnp.sum(
    (x - reconstructed_mean)**2 / variance + jnp.log(2 * jnp.pi * variance)
  )

def kl_divergence(μ, log_σ):
  return -0.5 * jnp.sum(1 + 2 * log_σ - μ**2 - jnp.exp(2 * log_σ))

def aevb_loss(params, x, rng_key):
  μ, log_σ = probabilistic_encoder(params['encoder'], x)
  z = reparameterize(rng_key, μ, log_σ)
  reconstructed_mean = decoder(params['decoder'], z)
  log_likelihood = gaussian_log_likelihood(x, reconstructed_mean)
  kl_loss = kl_divergence(μ, log_σ)
  return -log_likelihood + kl_loss


key = random.PRNGKey(0)
input_dim = 10
latent_dim = 5
hidden_dim = 8

key_params = random.split(key, 10)

encoder_params = {
  'W1': random.normal(key_params[0], (input_dim, hidden_dim)),
  'b1': jnp.zeros(hidden_dim),
  'W_μ': random.normal(key_params[1], (hidden_dim, latent_dim)),
  'b_μ': jnp.zeros(latent_dim),
  'W_σ': random.normal(key_params[2], (hidden_dim, latent_dim)),
  'b_σ': jnp.zeros(latent_dim)
}

decoder_params = {
  'W1': random.normal(key_params[3], (latent_dim, hidden_dim)),
  'b1': jnp.zeros(hidden_dim),
  'W_out': random.normal(key_params[4], (hidden_dim, input_dim)),
  'b_out': jnp.zeros(input_dim)
}

params = {'encoder': encoder_params, 'decoder': decoder_params}
single_data_point = random.normal(key_params[5], (input_dim,))
μ, log_sigma = probabilistic_encoder(params['encoder'], single_data_point)
z_sample = reparameterize(key_params[6], μ, log_sigma)
reconstructed = decoder(params['decoder'], z_sample)

𝓛 = aevb_loss(params, single_data_point, key_params[7])
```
