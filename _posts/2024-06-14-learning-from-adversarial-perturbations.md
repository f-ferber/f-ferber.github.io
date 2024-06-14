---
title: 'Learning from Adversarial Perturbations'
date: 2024-06-14
permalink: /posts/2024/06/learning-from-adversarial-perturbations/
tags:
  - machine learning theory
---

# This is a high-level summary of the paper

Kumano, Soichiro, Hiroshi Kera, and Toshihiko Yamasaki. **"Theoretical Understanding of Learning from Adversarial Perturbations."** arXiv preprint arXiv:2402.10470 (2024).

## What is this about?
Given a dataset, a binary classifier is trained. Then, an adversary perturbs (i.e., adds random noise to) every datapoint in a way such that the classifier would return the wrong label, thus generating a new dataset. Another classifier is trained from scratch on the perturbed and mislabeled dataset. It turns out that the second classifier can still generalize on the original, unperturbed test set. One hypothesis for this behavior is that the perturbation, although looking incomprehensible to humans, still has to contain some kind of label-data in order to fool the first classifier.

## The main theoretical result?
For some kind of structurally simple neural network (see below) and under the assumption that the training data points are mutually orthogonal (which they claim to be common in high-dimensional settings), the authors show the following: If the data points in the test set are neither strongly correlated with a few training samples nor weakly correlated with many training samples, then the adversarially trained classifier will produce the same classification as the original classifier with high probability.

## Which assumptions were made?
The neural network is assumed to have one hidden layer, and the weights $a \in \mathbb{R}^m$ of the output layer are frozen (not trainable) and set to $\frac{1}{\sqrt{m}}$ for the first half and $-\frac{1}{\sqrt{m}}$ for the second half of the weights from the hidden layer to the single output neuron. The activation function for the hidden layer is LeakyReLU $\phi(z) := \max(z, \gamma z)$. So if $W \in \mathbb{R}^{m \times d}$ denotes the weights for the hidden layer, the network can be written as $f(x) := a^\top \phi(Wx)$. This network is trained in the gradient flow regime (i.e., gradient descent with an infinitesimal step size) with either logistic or exponential loss. Several additional assumptions on the training data have to be made which relate the minimal norm of all training points $R_{\text{min}} := \min_{n} \Vert x_n \Vert$, the maximal norm of all training points $R_{\text{max}} := \max_n \Vert x_n \Vert$, the maximal absolute value of dot-products between all pairs of training points $p_{\text{max}} := \max_{i \neq j} \vert \langle x_i, x_j \rangle \vert$, and the LeakyReLU-parameter $\gamma$.

## How did they prove it?
The main ingredient for the proof is the fact that under the assumptions, the decision boundary of the neural network is a linear function. Concretely, assuming $\frac{\gamma^3 R_{\text{min}}^4}{3 N R_{\text{max}}^2} > p_{\text{max}}$, it holds $\text{sgn}(f(x)) = \text{sgn}(\tilde{f}(x))$, where $\tilde{f}$ is the linear function $\tilde{f}(x) := \sum_{i=1}^N \lambda_n y_n \langle x_n, x \rangle$, where $\lambda_n \in \left( \frac{1}{2 R_{\text{max}}^2}, \frac{3}{2 \gamma^2 R_{\text{min}}^2} \right)$ for every $n \in \{ 1, \dots, N \}$. Because of this, the adversarial perturbation of the training samples can be interpreted geometrically, showing that the added "noise" is, in fact, a linear combination of the training samples. The decision boundary of the adversarial classifier can be written as the sum of two terms, the first being related to the effect of mislabeled samples and the second being a function of the original decision boundary. If the second term outweighs the first one, the adversarial classifier can still generalize.

## Experimental results?
Experiments were run on a synthetic dataset for the one-hidden-layer setting and on MNIST, Fashion-MNIST, and CIFAR-10 for larger CNNs. The results show that learning from adversarial perturbations can achieve high test accuracy, even beyond the strict assumptions of their theory.
