---
layout: post
date: 2018-08-20 12:00:00 -0500
image: /data/2_knot_ReLU.jpg
---

## Bayes Theorem - The Basic Math

### Introduction

This article is meant to setup the mathematical foundation of Bayesian statistics. Step-wise modeling will be explained in another article.

### Set Notation

For discrete sets $A$ and $B$

$$P(A|B) = \frac{P(A \cap B)}{P(B)} \tag{1}$$

For binary A this simplifies to

$$P(A|B) = P(A \cap B)/(P(B|A)*P(A) + P(B|A)*P(A))$$

$$P(B|A) = \frac{P(A \cap B)}{P(A)} \implies P(A \cap B) = P(B|A) * P(A) \tag{2}$$

Substituting $2$ in $1$ we get

$$P(A|B) = \frac{P(B|A) * P(A)}{P(B|A) * P(A) + P(B|A') * P(A')}$$

Generalizing this to categorial $A$ with more than two levels:

$$P(A_j|B) = \frac{P(B|A_j) * P(A_j)}{\sum_{\forall i}P(B|A_i) * P(A_i)}$$

### Continuous $A$

Continuous form of Bayes theorem in the form of densities is given by:

$$P(A|B) = \frac{P(B|a) * f(a)}{\int P(B|a) * f(a) da}$$

### General Note

In commonly used statistical modeling methods such as GLM, we stop with $P(B\|A)$ as given by the data — this is the likelihood function. Bayesian modeling allows us to introduce prior beliefs about A into the system either through probability mass or through probability density function.

### Posterior Predictive Distribution

#### Definition

A posterior predictive distribution is the distribution of unobserved values conditioned on observed values.

#### Further Reading

The [Wikipedia page](https://en.wikipedia.org/wiki/Posterior_predictive_distribution) provides a rigorous treatment of posterior predictive distribution.

#### Mathematical Form

##### Unobserved Parameter

$$\pi(\theta|y) = \frac{f_{Y|\theta}(y|\theta) \pi(\theta)}{\int_{\Theta}f_{Y|\theta}(y|\theta) \pi(\theta)} \propto f_{Y|\theta}(y|\theta)\pi(\theta)$$

##### Unobserved Random Variable

$$f(y_2|\theta,y_1) = \int f(y_2|\theta,y_1)f(\theta|y_1)d\theta$$

If $y_2$ and $y_1$ are independent, this simplifies to $f(y_2\|\theta,y_1) = \int f(y_2\|\theta)f(\theta\|y_1)d\theta$

### Closing Notes

1. It is not always possible to obtain an analytical solution for the posterior predictive distribution.
2. In most practical cases where an analytical solution exists for the posterior predictive distribution, the denominator term is either equal to 1 or does not play a role in determining the type of posterior predictive distribution (this is not always true). Hence only the numerator is retained for further analysis.