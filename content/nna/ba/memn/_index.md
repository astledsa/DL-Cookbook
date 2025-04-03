---
title: "Memory Networks"
date: 2025-04-01
draft: false
references:
  - title: "Memory Networks"
    url: "https://arxiv.org/abs/1410.3916v10"
---

The idea of memory in deep learning models is an old one. Before modern methods took over (like RAG or long context lengthy models), researchers were coming up with various methods in order to instill memory in various types of networks. One such idea was the concept of memory networks. In retrospect, this idea is very similiar to LSTMs, but the approach is one of modularity and flexibility. The component agnostic method makes this approach quite interesting.

<br>

## Equation

\begin{align}
y=R(O(G(I(x), m), q))
\end{align}

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>R</strong></td>
    <td style="vertical-align: middle;">: Response component</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>O</strong></td>
    <td style="vertical-align: middle;">: Output feature map</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>G</strong></td>
    <td style="vertical-align: middle;">: Generalization component</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>I</strong></td>
    <td style="vertical-align: middle;">: Input feature map</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>m</strong></td>
    <td style="vertical-align: middle;">: Memory</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>q</strong></td>
    <td style="vertical-align: middle;">: Query</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output vector</td>
  </tr>
</table>

<br>

## Explanation

- Each component above has a fixed role, which seem very similar to the various gates in LSTM networks. The **input feature map** converts the query into an internal representatino (think embedding layers). The **generalization** component updates old memories, while taking into account the new information (allowing the network to compress information as well), could be the attention mechanism (more below).

- The **output feature map** produces a new output, taking into consideration the new information (query) and the memory index, in the internal representation space itself. And finally, **response** component converts the output back to a desired output space (analogous to the unembedding layer). 

- This setting lets us utilise *any* form for any component, with enough flexibility to mix and match various forms of input and output formats. I have decided to utilise a **vector index** as the memory component, **MLP** for input and ouput feature mapping and the infamous **attention mechanism** for the generalization component. Any other techniques could have been used (since the paper is from 2015, these methods weren't mainstream at the time)

<br>

## Code

```python
import jax
import hnswlib
from jax import nn
from jax import random
import jax.numpy as jnp

def initialize_params(key, input_dim, memory_dim, output_dim):
  k1, k2, k3, k4 = random.split(key, 4)
  params = {
      'input_mlp_w1': random.normal(k1, (input_dim, 16)),
      'input_mlp_b1': jnp.zeros(16),
      'input_mlp_w2': random.normal(k2, (16, memory_dim)),
      'input_mlp_b2': jnp.zeros(memory_dim),
      'gate_w': random.normal(k3, (memory_dim * 2, 1)),
      'gate_b': jnp.zeros(1),
      'response_mlp_w1': random.normal(k4, (memory_dim, 16)),
      'response_mlp_b1': jnp.zeros(16),
      'response_mlp_w2': random.normal(k4, (16, output_dim)),
      'response_mlp_b2': jnp.zeros(output_dim)
  }
  return params

def memory_network(m_hnsw, x, q, params):
    
  i_x = nn.relu(jnp.dot(x, params['input_mlp_w1']) + params['input_mlp_b1'])
  i_x = jnp.dot(i_x, params['input_mlp_w2']) + params['input_mlp_b2']

  labels, _ = m_hnsw.knn_query(np.array([i_x]), k=1)
  memory_vector = m_hnsw.get_items([labels[0][0]])[0]
  gate = nn.sigmoid(
      jnp.dot(
            jnp.concatenate(
                  [i_x, memory_vector]
            ), 
            params['gate_w']) + 
            params['gate_b']
      )
  updated_memory_vector = memory_vector * (1 - gate) + i_x * gate

  m_hnsw.mark_deleted(labels[0][0])
  m_hnsw.add_items(jnp.array([updated_memory_vector]), [labels[0][0]])

  labels, _ = m_hnsw.knn_query(jnp.array([q]), k=5)
  retrieved_memory = m_hnsw.get_items(labels[0])
  a = nn.softmax(jnp.dot(q, jnp.array(retrieved_memory).T))
  o = jnp.dot(a, jnp.array(retrieved_memory))

  y = nn.relu(jnp.dot(o, params['response_mlp_w1']) + params['response_mlp_b1'])
  y = jnp.dot(y, params['response_mlp_w2']) + params['response_mlp_b2']

  return y

input_dim = 5
output_dim = 5
memory_dim = 10
num_memory_slots = 10
key = random.PRNGKey(0)

m_hnsw = hnswlib.Index(space='l2', dim=memory_dim)
memory_vectors = jnp.array(random.normal(key, (num_memory_slots, memory_dim)))
m_hnsw.init_index(max_elements=num_memory_slots, ef_construction=200, M=16)
m_hnsw.add_items(memory_vectors, jnp.arange(num_memory_slots))

x = random.normal(key, (input_dim,))
q = random.normal(key, (memory_dim,))

y = memory_network_hnsw(
    m_hnsw, 
    x, 
    q, 
    initialize_params(
        key, 
        input_dim, 
        memory_dim, 
        output_dim
    )
  )

```