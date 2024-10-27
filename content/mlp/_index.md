---
title: "Multi-layer Perceptron"
date: 2023-04-20
draft: false
references:
  - title: "Grokking Deep learning"
    url: "https://edu.anarcho-copy.org/Algorithm/grokking-deep-learning.pdf"
  - title: "What is a neural network?"
    url: "https://youtu.be/aircAruvnKk?si=ZraEDT94knQqAs3M"
---

The central building block of any deep learning architecture. One of the main feature of MLPs, that essentially separeted it from classical Machine Learning was the use of hidden layers, which worked well with the gradient based optimization called backpropagation. 

## Equation

![MLP Diagram](/images/mlp.png)

<table style="border-collapse: collapse;">
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>x</strong></td>
    <td style="vertical-align: middle;">: Input matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W<sub>1</sub></strong></td>
    <td style="vertical-align: middle;">: First weight matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b<sub>1</sub></strong></td>
    <td style="vertical-align: middle;">: First bias vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>𝜎</strong></td>
    <td style="vertical-align: middle;">: Activation Function</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>W<sub>2</sub></strong></td>
    <td style="vertical-align: middle;">: Second weight matrix</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>b<sub>2</sub></strong></td>
    <td style="vertical-align: middle;">: Second bias vector</td>
  </tr>
  <tr>
    <td style="padding-right: 20px; vertical-align: middle;"><strong>y</strong></td>
    <td style="vertical-align: middle;">: Output vector</td>
  </tr>
</table>