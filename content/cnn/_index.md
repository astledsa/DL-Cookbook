---
title: "Convolutional Neural Network"
date: 2023-04-20
draft: false
references:
  - title: "Inductive Biases of CNNs"
    url: "https://g.co/gemini/share/e217804e5a2a"
  - title: "Mathematical perspective of CNNs"
    url: "https://mathblog.vercel.app/blog/cnn/"
  - title: "Theoretical analysis of CNNs"
    url: "https://arxiv.org/abs/2305.08404"
---

If not for the invention and eventual adoption of convolutional networks, deep learning as a field would never have taken off. Primarily used for image related tasks (everything from generation to segmentation), these models are still used in various tasks due to their favourable inductive bias towards images and an easily parallelisable architecture. 

<br>

## Equation

![CNN Diagram](/images/cnn2.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>x</i></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>K</strong><sub><i>p,s</i></sub></td>
    <td style="vertical-align: middle;">: Convolutional Kernel, with <i>p</i> padding and <i>s</i> stride</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>Pool</i></td>
    <td style="vertical-align: middle;">: Max/Average Pooling</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>w</i></td>
    <td style="vertical-align: middle;">: Width of the pooling window</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><i>h</i></td>
    <td style="vertical-align: middle;">: Height of the pooling window</td>
  </tr>
</table>

<br>

## Explanation

- The convolutional kernel filter provided the necessary methodology for the model to capture local patterns within the image, by learning a specific kernel (a parameter) that slides over the image. This follows the **locality principle**, where it is assumed that each pixel within the image is a much better predictor for it's immediate neighbours than for pixels much farther away. This assumption is an important one, and would be utilised extensively in other image-related model architectures.

- While the local patterns were being successfully captured by the filter, the global ones needed a different operation altogether. An object may appear multiple times in a single image, hence there's an invariance property that must be adhered to, namely **translation invariance**, which is what a pooling window accomplishes. It is a deterministic operation that "blurs" the images so that the model gets a bird's eye view of the entire image and thus can capture global patterns within the image. 

- The deep stacked architecture when coupled with layers of different sizes also led to a **hierarchical represenation** of an image, allowing the network to capture extremely complex features within an image, with alternating convolutional and pooling operations. This can be acheived by upsampling and downsampling of inputs in order for the model to learn different aspects of the image.

<br>

## Jax Code

```python
import jax
from jax import random
import jax.numpy as jnp

def convolution(x, k, s = 1, p = 'VALID'):
    if x.ndim == 2:
        x = x[None, :, :, None]
    elif x.ndim == 3:
        x = x[None, :]
    if k.ndim == 2:
        k = k[:, :, None, None]
    elif k.ndim == 3:
        k = k[:, :, :, None]
    if isinstance(s, int):
        s = (s, s)
    
    # The main convolution operation
    return jax.lax.conv(x, k, window_strides=s, padding=p)

def pooling(x, pool, s= None, T= 'max'):
    if x.ndim == 2:
        x = x[None, :, :, None]
    elif x.ndim == 3:
        x = x[None, :]
    if s is None:
        s = pool
    elif isinstance(s, int):
        s = (s, s)
    window_dims = (1,) + pool + (1,)
    s = (1,) + s + (1,)

    # The main pooling operations, max/average
    if T.lower() == 'max':
        return jax.lax.reduce_window( x, -jnp.inf, jax.lax.max, 
                                     window_dimensions=window_dims, 
                                      window_strides=s, 
                                      padding='VALID'
                                   )
    elif T.lower() == 'avg':
        sum_pool = jax.lax.reduce_window( x, 0., jax.lax.add, 
                                         window_dimensions=window_dims, 
                                          window_strides=s, 
                                          padding='VALID'
                                        )
        window_size = pool[0] * pool[1]
        return sum_pool / window_size
    else:
        raise ValueError("pool_type must be either 'max' or 'avg'")

 # Initialise a random key
key = random.PRNGKey(100)
key1, key2 = random.split(key, num=2)

# A random vector as an input (in_channels, image_width, image_height)
x  = random.normal(key1, (3, 225, 225))

# Kernel K (out_channels, in_channels, kernel_width, kernel_height)
k = random.normal(key2, (3, 3, 5, 5))

# Output
y = pooling(convolution(x, k), (2, 2), 2, 'avg')
```
*Written by Claude Sonnet 3.5, for more details on jax.lax.conv visit [here](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L56), and a simplified overview of the convolution operation is [here](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#the-cross-correlation-operation)* 