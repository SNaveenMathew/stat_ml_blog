---
layout: post
date: 2018-09-17 12:00:00 -0500
image: /data/2_knot_ReLU.jpg
---

## Introduction to Bayesian Statistics - Part 2

### Introduction

This article is intended to provide more clarity on the differences in approach between frequentist and Bayesian methods. It will introduce the basics of Bayesian statistics that will help in understanding the first article ([Introduction to Bayesian statistics — part 1](https://snaveenmathew.github.io/stat_ml_blog/2018/07/18/Introduction-to-Bayes-Statistics.html)).

### Frequentist Approach

The frequentist approach assumes that the underlying parameters are constant and unknown. In other words, the sample that we observed is from a distribution with constant parameters. The sample can be used to estimate the value of the parameters by optimizing a suitable error metric that's related to the underlying distribution.

For example, if we know that the observations $x_1, x_2, ..., x_n \sim N(M, \sigma^2)$ with known $\sigma^2$, we can estimate M by optimizing the likelihood function $L(X\|\mu) = \prod_{i=1}^{n}\frac{e^{-\frac{(x-\mu)^2}{2\sigma^2}}}{\sqrt{2\pi\sigma^2}}$. Maximimizing the likelihood (or minimizing the negative log-likelihood ~ ordinary least squares) leads to the estimator $\mu = \frac{\sum_{i=1}^{n} x_i}{n}$.

#### Frequentist Inference

Let us assume that the sample estimator is $\mu$ and sample standard error of the estimation is $s=\frac{\sigma}{\sqrt{n}}$. We can build the $(1-\alpha)\%$ confidence interval $\[μ - z_α * s, μ + z_α * s\]$.

However, this does not mean that there is 95% probability of $Μ$ lying in this particular interval.

#### Interpretation of Frequentist Confidence Interval

If we keep collecting multiple samples and create confidence intervals, we expect 95% of the confidence intervals to contain $M$.

### Bayesian Approach

The idea of an underlying constant, unknown parameter limits the type of inference that can be performed using frequentist approach. In the example of normal distribution with constant mean $M$ and constant, known variance $\sigma^2$, we cannot make an inference with $H_0: M=0$ vs. $H_1: M \neq 0$. By allowing the underlying parameter to be a random variable, the Bayesian approach allows us to make a guess for the underlying parameter (prior distribution) and to update the guess based on the observed data (likelihood * prior = posterior).

#### Inference

We can start by assuming that Μ follows a distribution with a certain pdf/pmf. This is our prior distribution. After observing the data, we may ask the following question:

1. Can we update our prior knowledge of Μ based on the data that was observed to accurately represent the data generating process?
2. Can we infer something about Μ? For example: $H_0: M=c$ vs. $H_1: M \neq c$

Bayesian statistics allows us to answer both questions in a reasonable way. However, the quality of inference can be greatly affected by the choice of prior for the parameters. It is safer to use ‘non-informative’ priors, but this is not always the best strategy.