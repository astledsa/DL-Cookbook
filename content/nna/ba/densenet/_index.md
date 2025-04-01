---
title: "DenseNet"
date: 2025-03-31
draft: false
references:
  - title: "Densely Connected Convolutional Networks"
    url: "https://paperswithcode.com/paper/densely-connected-convolutional-networks"
---

Following the success of convolutional networks, the field of computer vision saw it's fair share of diverse innovations and architecture changes, all in hopes of doing what AlexNet once accomplished. One such interesting architecure is the DenseNet, which relies heavily on feature reuse, and takes the idea of residual networks one step further in a manner.

<br>

## Equation

\begin{align}
y=W_o(Pool(\phi(\sigma(W^{(1x1)}(D(Conv_{7x7}(x)))))))
\end{align}

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>D</i></td>
    <td style="vertical-align: middle;">: The dense block, (H<sub>l</sub>([x<sub>0</sub>, x<sub>1</sub>, ..., x<sub>l-1</sub>]))</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>H<sub>l</sub></i></td>
    <td style="vertical-align: middle;">: The dense convoluton function, where each input is concatenated with it's output</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>ϕ</strong></td>
    <td style="vertical-align: middle;">: Batch Normalization function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>σ</strong></td>
    <td style="vertical-align: middle;">: ReLU activation function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>Pool</i></td>
    <td style="vertical-align: middle;">: Average Pooling layer</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>Conv</i></td>
    <td style="vertical-align: middle;">: Convolutional layer</td>
  </tr>
</table>

<br>

## Explanation

- As is mentioned in the above legend, the main contribution of this paper is the *dense* block, which follows a simple principle: concatenate the previous inputs to the input of the *l*th layer in order for the function to re-use learned features. While this is a simple modification over the ResNet architecture, it proved quite effective in experimentations (refer to the paper). 

- The bottleneck (W<sup>(1x1)</sup>x) and the transition layers (BN(ReLU(x))) were added for efficiency, where the bottle neck layer reduced dimensions of the inputs and the transition layer reduces the channel and spatial dimensions (which meant lower dimensional weights for initialization)

- One of the advantages was that this made the model quite **parameter efficient**, since fewer parameters were needed due to the **feature reuse**, as each layer was provided with the information of previous layer directly, unlike in ResNets. This ended up making the model compact, without much loss and in fact a gain in accuracy. 

- Another aspect that was mentioned in the paper was the ability of **deep supervision**, in the words of the authors:<br> *"One explanation for the improved accuracy of dense convolutional networks may be
that individual layers receive additional supervision from
the loss function through the shorter connections. One can
interpret DenseNets to perform a kind of “deep supervision”."*

<br>

## Code

```python
import jax
from jax import random
import flax.linen as nn
import jax.numpy as jnp

# ---------------------------------------------------- #
# Simpler Implementation
# ---------------------------------------------------- #

def batch_norm(X, epsilon=1e-5):
    gamma = jnp.ones(X.shape[-1])
    beta = jnp.zeros(X.shape[-1])

    mean = jnp.mean(X, axis=0, keepdims=True) 
    variance = jnp.var(X, axis=0, keepdims=True)
    X_normalized = (X - mean) / jnp.sqrt(variance + epsilon)
    out = gamma * X_normalized + beta
    return out

def avg_pool(x, window_shape=(2, 2), strides=(2, 2)):
    N, H, W, C = x.shape
    window_h, window_w = window_shape
    stride_h, stride_w = strides
    
    out_h = (H - window_h) // stride_h + 1
    out_w = (W - window_w) // stride_w + 1
    
    patches = jnp.reshape(
        x[:, :out_h * stride_h, :out_w * stride_w, :],
        (N, out_h, stride_h, out_w, stride_w, C)
    )
    patches = patches.transpose(0, 1, 3, 2, 4, 5) 
    pooled = jnp.mean(patches, axis=(3, 4))
    
    return pooled

def conv(x, num_filters=32, filter_size=(3, 3), key=random.PRNGKey(0)):
    N, H, W, C = x.shape
    filter_h, filter_w = filter_size

    key1, key2 = random.split(key)
    weights_shape = (num_filters, C, filter_h, filter_w)
    fan_in = C * filter_h * filter_w
    fan_out = num_filters * filter_h * filter_w
    stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
    weights = random.normal(key1, weights_shape) * stddev
    
    bias = jnp.zeros((num_filters,))
    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant')
    out = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=weights,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC')
    )

    out = out + bias[None, None, None, :]
    return out

def Dense (x, n, growth_rate) :
  for _ in range(n):
    out = batch_norm(x)
    out = jax.nn.relu(out)
    out = conv(x)
    jnp.concatenate([x, out], axis=-1)
  return out

key = random.PRNGKey(0)
x = random.normal(key, (1, 28, 28, 64))
b = random.normal(key, (64, 1))
W = random.normal(key, (64, 32))

y = W @ (jnp.mean(
      avg_pool(conv(jax.nn.relu(batch_norm(
          Dense(
              conv(x), 3, 32
          )
      )))), axis=(1,2)
    )).T + b


# ---------------------------------------------------- #
# Proper Implementation
# ---------------------------------------------------- #

class DenseLayer(nn.Module):
    growth_rate: int

    @nn.compact
    def __call__(self, x):
        out = nn.BatchNorm(use_running_average=False)(x)
        out = nn.relu(out)
        out = nn.Conv(features=self.growth_rate, kernel_size=(3, 3), padding='SAME')(out)
        return jnp.concatenate([x, out], axis=-1)

class DenseBlock(nn.Module):
    num_layers: int
    growth_rate: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = DenseLayer(self.growth_rate)(x)
        return x

class TransitionLayer(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

class MiniDenseNet(nn.Module):
    num_classes: int = 10
    growth_rate: int = 12
    num_layers_per_block: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = DenseBlock(self.num_layers_per_block, self.growth_rate)(x)
        x = TransitionLayer(out_channels=32)(x)
        x = DenseBlock(self.num_layers_per_block, self.growth_rate)(x)
        x = TransitionLayer(out_channels=64)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=self.num_classes)(x)
        return x


model = MiniDenseNet(num_classes=10, growth_rate=12, num_layers_per_block=3)
rng = jax.random.PRNGKey(0)
x = jnp.ones((1, 32, 32, 3))
params = model.init(rng, x)
y = model.apply(params, x, mutable=['batch_stats'])[0]
```

***NOTE**: The crude implementation only takes one from the input to the output, and hence initialises the parameters within the function. This is simply for reference and understanding. The proper implementation uses Flax in order to track the parameters automatically, since it gets tedious to implement the tracking crudely.*