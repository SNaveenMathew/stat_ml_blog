## Introduction to Bayesian Statistics - Part 2

### Introduction

This article is intended to provide more clarity on the differences in approach between frequentist and Bayesian methods. It will introduce the basics of Bayesian statistics that will help in understanding the first article ([Introduction to Bayesian statistics — part 1](https://snaveenmathew.github.io/stat_ml_blog/2018/07/18/Introduction-to-Bayes-Statistics.html)).

### Frequentist Approach

The frequentist approach assumes that the underlying parameters are constant and unknown. In other words, the sample that we observed is from a distribution with constant parameters. The sample can be used to estimate the value of the parameters by optimizing a suitable error metric that's related to the underlying distribution.

For example, if we know that the observations $x_1, x_2, ..., x_n \sim N(M, \sigma^2)$ with known $\sigma^2$, we can estimate M by optimizing the likelihood function $L(X\|\mu) = \prod_{i=1}^{n}\frac{e^{-\frac{(x-\mu)^2}{2\sigma^2}}}{\sqrt{2\pi\sigma^2}}$. Maximimizing the likelihood (or minimizing the log-likelihood ~ ordinary least squares) leads to the estimator $\mu = \frac{\sum_{i=1}^{n} x_i}{n}$.

#### Frequentist Inference

Let us assume that the sample estimator is $\mu$ and sample standard error of the estimation is $s=\frac{\sigma}{\sqrt{n}}$. We can build the $(1-\alpha)\%$ confidence interval $\[μ - z_α * s, μ + z_α * s\]$.