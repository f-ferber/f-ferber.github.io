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

For linear neural networks with quadratic loss, there exist two critical step sizes \\( 0 < \eta_C < \eta_E \\) such that:

- For every step size \\( \eta > \eta_C \\) there are no flat minima along the optimization trajectory in which gradient descent can get "stuck"
- For every step size \\( \eta > \eta_E \\) gradient descent only converges with probability zero.

The region \\( [\eta_C, \eta_E] \\) defines the Edge of Stability, where convergence is possible but can't get trapped into flat critical points.

## Linear networks?

A linear neural network is just a function \\( \Phi(x; \theta) := W(\theta) \cdot x \\) for \\( x \in \mathbb{R}^{d_x} \\) and \\( W(\theta) \in \mathbb{R}^{d_x \times d_y} \\). The map \\( W: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_x \times d_y} \\) with \\( d_\theta := d_x \cdot d_y \\) just rearranges the vector \\( \theta \\) into a \\( d_x \times d_y \\)-matrix. Notice that the authors use a formalization of their linear neural networks with several layers, which is deceptive because all layers can be collapsed into one by just matrix-multiplication. The loss function is assumed to be the quadratic loss \\( L(\theta) := \sum_{i=1}^n \Vert y_i - \Phi(x_i; \theta) \Vert^2 \\), where \\( { (x_i, y_i) }_{i=1}^n \\) is the training set.

## Proof idea for the first result?

Gradient descent can be seen as the iterative application of the gradient descent map \\( G: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta}, \quad \theta \mapsto \theta - \eta \nabla L(\theta) \\) for step-size \\( \eta \\). Notice that every critical point \\( \overline{\theta} \\) of the loss landscape is a fixed point of \\( G \\), i.e., \\( G(\overline{\theta}) = \overline{\theta} \\).

For a smooth mapping \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\), a fixed point \\( \overline{\theta} \\) is called _Lyapunov stable_ if for every \\( \epsilon > 0 \\) there exists a \\( \delta > 0 \\) such that for every point \\( x \\) in the \\( \delta \\)-ball around the fixed point \\( \overline{\theta} \\), all points \\( f(x), f^2(x), f^3(x), \dots \\) stay within an \\( \epsilon \\)-ball of \\( \overline{\theta} \\), where we denote by \\( f^n \\) the \\( n \\)-th iterated application of \\( f \\), i.e. \\( f(f(\dots f(x) \dots)) \\). All other fixed points are called _Lyapunov unstable_.

Using this definition, one can partition the set of fixed points of the gradient descent map \\( G \\) (that is, the set of critical points of the loss landscape) into stable and weakly stable critical points.

Let \\( \lambda_i: M \to \mathbb R \\) be the function that maps any minimizer \\( \theta \in M \\) to the \\( i \\)-th non-zero eigenvalue of the Hessian \\( H_L(\theta) \\) of the loss function \\( L \\) at \\( \theta \\). Denote by \\( M \\) the set of minimizers for the linear neural network, which can be shown to be a manifold. For every step size \\( \eta \\), the strong stable sub-manifold is defined as \\( M_{\text{SS}} := \left\\{ \theta \in M \; \left\lvert \; \lambda_\text{max}(\theta) < \frac{2}{\eta} \right. \right\\} \\), where \\( \lambda_\text{max}(\theta) \\) denotes the maximal eigenvalue of the Hessian of \\( G \\) at \\( \theta \\). Let \\( S(G) := \left\\{ \left. \theta \in \mathbb{R}^{d_\theta} \; \right\rvert \; \det(\text{d}G(\theta)) = 0 \right\\} \\) be the set of singularities of \\( G \\).

From the definition of \\( G \\), we can see that its total derivative is given by \\( \text{d}G(\theta) = I - \eta H_L(\theta) \\). If the step size \\( \eta \\) is smaller than the largest eigenvalue of the Hessian along the optimization trajectory, then \\( \text{d}G \\) has an eigenvalue less than one, meaning that \\( G \\) is contractive and thus gradient descent is guaranteed to converge.

Hence, it is not surprising that all points in \\( M_\text{SS} \setminus S(G) \\) are Lyapunov stable, while every other point in \\( M \\) is unstable.

The authors show the important technical result that \\( \lambda_i \\) is a _proper function_ for every \\( i \\), i.e. that for any compact set \\( K \subseteq \mathbb R \\) its preimage \\( \lambda_i^{-1}(K) \subseteq M \\) is also compact. As a consequence, there exists an \\( \eta > 0 \\) such that \\( \lambda_i^{-1} \\left( \\left[ 0, \frac{1}{\eta} \\right] \\right) = \emptyset \\) for every non-zero \\( i \\)-th eigenvalue of the Hessian of \\( L \\).

Using the previous result, one can determine the critical step size \\( \eta_C \\) such that \\( \lambda_\text{max}^{-1} \\left( \\left[ 0, \frac{1}{\eta_C} \\right] \\right) = \emptyset \\), where \\( \lambda_\text{max} \\) refers to the maximum eigenvalue of \\( H_L \\). Consequently, for gradient descent with larger step sizes, \\( M_\text{SS} \\) will be empty, i.e. have measure zero.

A map \\( f: \mathbb{R}^{d_\theta} \to \mathbb{R}^{d_\theta} \\) is called non-singular if for any \\( B \subseteq \mathbb{R}^{d_\theta} \\) with measure zero, its preimage \\( f^{-1}(B) \\) also has measure zero. One of the main technical contributions of the paper is a proof that the gradient descent map is non-singular for linear networks with quadratic loss. This allows them to transform a local property into a global property: If all convergent trajectories land in a set of measure zero (like \\( M_\text{SS} \\)), then also the preimages of iterated gradient descent steps have measure zero. So convergence into this set (of flat minimizers) doesn't happen almost surely.

Define \\( C \subseteq \mathbb{R}^{d_\theta} \\) as the set of parameters from which gradient descent will converge. If \\( M_\text{SS} \\) has non-zero measure, by the definition of Lyapunov-stability, there must be an \\( \epsilon \\)-ball around every minimum that is also a subset of \\( C \\). Because this \\( \epsilon \\)-ball has non-zero measure, \\( C \\) must have non-zero measure too. So convergence to a Lyapunov unstable minimizer is still possible.

As a consequence, for \\( \eta > \eta_C \\), convergence is possible, but guaranteed to not get "trapped" into a Lyapunov stable critical point.

## Proof idea for the second result?

Define \\( M_{\text{WS}} := \left\\{ \theta \in M \; \left\lvert \; \lambda_\text{min}(H(\theta)) < 2\eta \right. \right\\} \\) as the submanifold of _weakly stable_ minimizers. With a similar argument as before, there exists a critical step size \\( \eta_E \\) such that \\( M_\text{WS} \neq \emptyset \\).

But now the authors show that \\( C \\) has measure zero. Because \\( G \\) is non-singular, the set of initial weights that converge has measure zero, i.e., convergence is almost surely impossible.

## Final thoughts?

I like this paper very much, because it uses topological arguments (non-singular maps, proper functions, ...) to shed light onto the geometry of the loss landscape as seen from the perspective of gradient descent.
