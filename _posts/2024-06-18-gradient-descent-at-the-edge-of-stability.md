---
title: 'Learning at the Edge of Stability'
date: 2024-06-18
permalink: /posts/2024/06/learning-at-the-edge-of-stability/
tags:
  - machine learning theory
---

Gradient descent is a cornerstone algorithm in training neural networks. However, using large learning rates often leads to instability. Interestingly, recent studies suggest that training in this unstable regime — known as the _Edge of Stability_ — can lead to models that generalize better. This blog post summarizes a paper/preprint that offers a theoretical justification for this phenomenon.

This is a high-level summary of the paper

Crăciun, Alexandru, and Debarghya Ghoshdastidar. "On the Stability of Gradient Descent for Large Learning Rate." arXiv preprint [arXiv:2402.13108](https://arxiv.org/abs/2402.13108) (2024).

## The main result?

For linear neural networks with quadratic loss, there exists an interval \\( ]\eta_C, \eta_E] \subseteq \mathbb R \\) such that for every step size \\( \eta \\) within this interval, the gradient descent algorithm will converge with probability one (over the choice of the random initial weights) towards a flat minimizer.

## Linear networks?

A linear neural network is just a function \\( \Phi(x; \theta) := W(\theta) \cdot x \\) for \\( x \in \mathbb{R}^{d_x} \\) and \\( W(\theta) \in \mathbb{R}^{d_x \times d_y} \\). The map \\( W: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_x \times d_y} \\) with \\( d_\theta := d_x \cdot d_y \\) just rearranges the vector \\( \theta \\) into a \\( d_x \times d_y \\)-matrix. Notice that the authors use a formalization of their linear neural networks with several layers, which is deceptive because all layers can be collapsed into one by just matrix-multiplication. The loss function is assumed to be the quadratic loss \\( L(\theta) := \sum_{i=1}^n \Vert y_i - \Phi(x_i; \theta) \Vert^2 \\), where \\( { (x_i, y_i) }_{i=1}^n \\) is the training set.

## Proof idea?

Gradient descent can be seen as the iterative application of the gradient descent map \\( G: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta}, \quad \theta \mapsto \theta - \eta \nabla L(\theta) \\) for step-size \\( \eta \\). Notice that every critical point \\( \overline{\theta} \\) of the loss landscape is a fixed point of \\( G \\), i.e., \\( G(\overline{\theta}) = \overline{\theta} \\).

A map \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\) is called non-singular if for any \\( B \subseteq \mathbb{R}^{d_\theta} \\) with measure zero, its preimage \\( f^{-1}(B) \\) also has measure zero. One of the main technical contributions of the paper is a proof that the gradient descent map is non-singular for linear networks with quadratic loss. This allows them to transform a local property into a global property: If all convergent trajectories land in a set of measure zero, then also the preimages of iterated gradient descent steps have measure zero. So convergence into this set doesn't happen almost surely.

For a smooth mapping \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\), a fixed point \\( \overline{\theta} \\) is called Lyapunov stable if for every \\( \epsilon > 0 \\) there exists a \\( \delta > 0 \\) such that for every point \\( x \\) in the \\( \delta \\)-ball around the fixed point \\( \overline{\theta} \\), all points \\( f(x), f^2(x), f^3(x), \dots \\) stay within an \\( \epsilon \\)-ball of \\( \overline{\theta} \\). All other fixed points are called Lyapunov unstable.

Using this definition, one can partition the set of fixed points of the gradient descent map \\( G \\) (that is, the set of critical points of the loss landscape) into stable and weakly stable critical points. If the step size is chosen too large, then the set of both stable and weakly stable minima has measure zero, and thus convergence becomes impossible almost surely. If the step size is small enough, however, the set of stable minima has zero measure, but the set of weakly stable minima has non-zero measure and thus optimization enters the Edge of Stability regime, attending a weakly stable minimum.

## Any more details?

From the definition of \\( G \\), we can see that its total derivative is given by \\( \text{d}G(\theta) = I - \eta H(\theta) \\), where \\( H \\) is the Hessian of \\( G \\). If the step size \\( \eta \\) is smaller than the largest eigenvalue of the Hessian along the optimization trajectory, then \\( \text{d}G \\) has an eigenvalue less than one, meaning that \\( G \\) is contractive and thus gradient descent is guaranteed to converge.

Denote by \\( M \\) the set of minimizers for the linear neural network, which can be shown to be a manifold. For every step size \\( \eta \\), the strong stable sub-manifold is defined as \\( M_{\text{SS}} := \left{ \theta \in M ; \vert ; \lambda_\text{max}(H(\theta)) < \frac{2}{\eta} \right} \\), where \\( \lambda_\text{max}(H(\theta)) \\) denotes the maximal eigenvalue of the Hessian of \\( G \\) at \\( \theta \\). Let \\( S(G) := \left{ \theta \in \mathbb{R}^{d_\theta} ; \vert ; \det(\text{d}G(\theta)) = 0 \right} \\) be the set of singularities of \\( G \\). By the above discussion, it is not surprising that all points in \\( M_\text{SS} \setminus S(G) \\) are Lyapunov stable, while every other point in \\( M \\) is unstable.

Define \\( C \subseteq \mathbb{R}^{d_\theta} \\) as the set of parameters from which gradient descent will converge. If \\( M_\text{SS} \\) has non-zero measure, by the definition of Lyapunov-stability, there must be an \\( \epsilon \\)-ball around every minimum that is also a subset of \\( C \\). Because this \\( \epsilon \\)-ball has non-zero measure, \\( C \\) must have non-zero measure too. So convergence to a stable minimizer is possible.

The authors show also the opposite: If the step size is too large, then \\( C \\) will have measure zero, so convergence is almost surely impossible.

## Final thoughts?

In summary, this paper sheds light on why gradient descent with large learning rates can still converge and even lead to better generalization in neural networks. By understanding the stability properties and behavior of the gradient descent map, we gain valuable insights into the dynamics of optimization in machine learning.
