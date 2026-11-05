---
layout: post
date: 2026-11-04 12:00:00 -0000
image: /data/NN_design1.jpg
---

## Deeper analysis of bias constraints and loss surface of a sigmoid network

In the [sigmoid-network practicum](https://github.com/SNaveenMathew/stat_ml_blog/blob/master/_posts/2022-08-14-Sigmoid-Network-Practicum.md), I compared two small networks trained to approximate $y=3x^2+2x+1$. The bias-constrained network looked more predictable outside the training range, while the unconstrained network followed the training data more closely. I wanted to see what the loss looked like between those two sets of weights.

I generated interactive three-dimensional slices through the loss surface around the stored constrained vector. Each slice varies two parameters and holds the other five at their constrained values. They show how loss changes in selected directions; they do not establish that fixing biases generally improves extrapolation.

### Model and data

I used one input, two sigmoid units, and a linear output:

$$h_j(x)=\sigma(w_jx+b_j), \qquad \hat y=B_0+W_1h_1(x)+W_2h_2(x)$$

I generated 10,000 inputs from $N(0,1)$ and set the target to $y=3x^2+2x+1$. In the constrained reference, $b_1=0$ and $b_2=\log(2-\sqrt{3})\approx-1.317$ stay fixed. Both hidden biases are free in the unconstrained reference. The script stores seven parameters for each vector in this order:

$$(w_1,w_2,b_1,b_2,W_1,W_2,B_0)$$

$$\theta_{unconstrained}=(-1.0867,\ 0.8551,\ -3.1022,\ -2.7769,\ 51.5640,\ 91.1212,\ -6.4820)$$

$$\theta_{constrained}=(0.1900,\ 0.3927,\ 0,\ -1.3170,\ -559.892,\ 434.2152,\ 189.08543)$$

For the second panel, I multiply the same inputs by 3 and recompute the quadratic target. That lets me see how the vectors behave over a wider range. Since the inputs are reused, this is not an independent validation or test sample.

### Three-dimensional loss slices

Each interactive plot varies one pair of parameters over a grid and holds the other five at their constrained values. The left surface shows training loss; the right shows loss on the same inputs scaled by 3, with the quadratic target recomputed. Both surfaces use $\log_{10}(\mathrm{MSE})$ on a shared color scale. The red marker is the constrained pair of values. The cyan marker substitutes the unconstrained values for just the two displayed parameters; it is not the loss of the complete unconstrained model.

The hidden-kernel slice shows a broad, curved basin with steep walls, while the output-weight slice has a particularly narrow valley. The hidden-bias slice is sharply curved as well. These ridges and valleys show that the loss is not a simple convex bowl in these parameter pairs. Since the plots are two-dimensional slices through a seven-dimensional function, they cannot establish global non-convexity on their own or reveal every direction in which loss can change.

<figure>
  <iframe src="../../../data/loss_surfaces_3d/loss_surface_3d_hidden_kernel_1_vs_hidden_kernel_2.html" title="Interactive loss surface for the two hidden-layer weights" width="100%" height="700" loading="lazy"></iframe>
  <figcaption>Hidden-layer weights: a broad curved basin and steep changes in loss. Rotate either panel to explore the surface.</figcaption>
</figure>

<figure>
  <iframe src="../../../data/loss_surfaces_3d/loss_surface_3d_output_weight_1_vs_output_weight_2.html" title="Interactive loss surface for the two output weights" width="100%" height="700" loading="lazy"></iframe>
  <figcaption>Output weights: a narrow valley illustrates strong sensitivity to their joint values.</figcaption>
</figure>

<figure>
  <iframe src="../../../data/loss_surfaces_3d/loss_surface_3d_hidden_bias_1_vs_hidden_bias_2.html" title="Interactive loss surface for the two hidden-layer biases" width="100%" height="700" loading="lazy"></iframe>
  <figcaption>Hidden-layer biases: changing these parameters moves away from the constrained model's feasible set.</figcaption>
</figure>

### Takeaways

The red constrained marker has lower loss on the scaled-input panel than the cyan hybrid marker in each of these slices. This suggests that the constrained fit is less sensitive to this particular input-range expansion around the plotted parameter pairs. It does not show that the complete constrained model is more robust than the complete unconstrained model: each cyan marker combines two unconstrained parameter values with the other five constrained values. A fair comparison would evaluate both full fitted vectors on the same independent, wider-range test set. Other parameters, including the output intercept, differ between the stored fits, so these slices cannot isolate the effect of the bias constraint.

A weighted sum of sigmoid units approaches a constant at either end of the real line, while the target quadratic keeps growing. Neither version of this network can match the target indefinitely, whatever the biases are.

The scaled-input panel is a stress test on reused samples, not independent validation. It cannot establish which full model extrapolates better on new data or whether the bias constraint caused a difference. That would require independent samples and controlled fits that differ only in the constraint.
