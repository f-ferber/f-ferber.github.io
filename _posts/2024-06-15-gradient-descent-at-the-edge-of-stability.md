---
title: 'Learning at the Edge of Stability'
date: 2024-06-14
permalink: /posts/2024/06/learning-at-the-edge-of-stability/
tags:
  - machine learning theory
---

It has been observed that training neural networks by gradient descent with step sizes larger than the stable one (the "Edge of Stability" regime) can converge (albeit non-monotonically) to minimizers that generalize better. This post summarizes a paper/preprint that gives a theoretical justification for this phenomenon.

This is a high-level summary of the paper

Crăciun, Alexandru, and Debarghya Ghoshdastidar. "On the Stability of Gradient Descent for Large Learning Rate." arXiv preprint [arXiv:2402.13108](https://arxiv.org/abs/2402.13108) (2024).

## What is this about?
Gradient descent is guaranteed to work ("Descent Lemma") when the loss function has Lipschitz continuous gradient and the step size (learning rate) stays strictly smaller than the inverse of the largest eigenvalue of the Hessian of the loss along the optimization trajectory. Training with larger step sizes can still converge, albeit non-monotonically. This is called the _Edge of Stability (EoS)_ regime and is not fully understood yet. The authors give a theoretical justification for the case of very simple neural networks.

## The main result?
For linear neural networks with quadratic loss there exists an interval \\( ]\eta_C, \eta_E] \subseteq \mathbb R \\) such that for every step size \\( \eta \\) within this interval the gradient descent algorithm will converge with probability one (over the choice of the random initial weights) towards a flat minimizer.

## Linear networks?
A linear neural network is just a function \\( \Phi(x; \theta) := W(\theta) \cdot x \\) for \\( x \in \mathbb{R}^{d_x} \\) and \\( W(\theta) \in \mathbb{R}^{d_x \times d_y} \\). The map \\( W: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_x \times d_y} \\) with \\( d_\theta := d_x \cdot d_y \\) just rearranges the vector \\( \theta \\) into a \\( d_x \times d_y \\)-matrix. Notice that the authors use a formalization of their linear neural networks with several layers, which is deceptive, because all layers can be collapsed into one by just matrix-multiplication. The loss function is assumed to be the quadratic loss \\( L(\theta) := \sum_{i=1}^n \Vert y_i - \Phi(x_i; \theta) \Vert^2 \\), where \\( \{ (x_i, y_i) \}_{i=1}^n \\) is the training set.

## Proof idea?
Gradient descent can be seen as the iterative application of the _gradient descent map_ \\( G: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta}, \quad \theta \mapsto \theta - \eta \nabla L(\theta) \\) for step-size \\( \eta \\). Notice that every critical point \\( \overline{\theta} \\) of the loss landscape is a fixed point of \\( G \\), i.e. \\( G(\overline{\theta}) = \overline{\theta} \\).

A map \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\) is called _non-singular_ if for any \\( B \subseteq \mathbb{R}^{d_\theta} \\) with measure zero its preimage \\( f^{-1}(B) \\) has also measure zero. One of the main technical contributions of the paper is a proof that the gradient descent map is non-singular for linear networks with quadratic loss. This allows them to transform a local property into a global property: If all convergent trajectories land in a set of measure zero, then also the preimages of iterated gradient descent steps have measure zero. So convergence into this set doesn't happen almost surely.

For a smooth mapping \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\), a fixed point \\( \overline{\theta} \\) is called _Lyapunov stable_ if for every \\( \epsilon > 0 \\) there exists a \\( \delta > 0 \\) such that for every point \\( x \\) in the \\( \delta \\)-ball around the fixed point \\( \overline{\theta} \\), all points \\( f(x), f^2(x), f^3(x), \dots \\) stay within an \\( \epsilon \\)-ball of \\( \overline{\theta} \\). All other fixed points are called _Lyapunov unstable_.

Using this definition, one can partition the set of fixed points of the gradient of the gradient descent map \\( G \\) (that is, the set of critical points of the loss landscape) into stable and weakly stable critical points. If the step size is chosen too large, then the set of both, stable and weakly stable minima, has measure zero and thus convergence becomes impossible almost surely. If the step size is small enough, however, the set of stable minima has zero measure, but the set of weakly stable minima has non-zero measure and thus optimization enters the Edge of Stability regime, attending a weakly stable minimum.
