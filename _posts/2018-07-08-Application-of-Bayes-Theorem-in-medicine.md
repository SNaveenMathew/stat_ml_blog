---
layout: post
date: 2018-07-08 12:00:00 -0500
image: ../../../data/regressionfit.gif
---

## Applications of Bayes Theorem in Medicine

### Introduction

Medicine has rightly embraced the scientific method of "put up or shut up" (~burden of proof). Statistical inference on carefully chosen test-control samples has been one of the key drivers of progress in drug testing. In this article we will study the effect of choice of sample on the effectiveness of a test from a statistician's point of view. We will use discrete form of Bayes theorem defined by:

$P(X \| A) = P(A \| X) * P(X)/\sum_{\forall i}(P(A \| Y_i) * P(Y_i))$ for exhaustive set of outcomes $Y_i \in Y$, where X is an outcome in the discrete set $Y$.

---

### Setting 1

Let us assume that we have designed a test for a disease X. The test results belong to the set $\\{+, -\\}$.

Let $P(+ \| X) = 0.99$, $P(- \| no X) = 0.98$, $P(X) = 0.005$.

If a person has disease X, he/she is 99% likely to be detected as positive (correctly) by the test. If a person does not have disease X, he/she is 98% likely to be detected as negative (correctly) by the test. First look suggests that the test is very effective. However, let’s take a closer look.

---

### Evaluation of Setting 1

$$P(X | +) = \frac{P(+ | X) * P(X)}{(P(+ | X) * P(X) + P(+ | no X) * P(no X))} = \frac{0.99*0.005}{(0.99*0.005 + 0.02*0.995)} = 0.199$$

This is not good.

$$P(no X | -) = \frac{P(- | no X) * P(no X)}{(P(- | no X) * P(no X) + P(- | X) * P(X))} = \frac{0.98*0.995}{(0.98*0.995 + 0.01*0.005)} = 0.9999$$

This is good.

If the test predicts that the person suffers from disease X, there is only 20% chance that the person actually has disease X. This is not a good test even though there is a clear 'lift' in performance compared to a random guess.

---

### Setting 2

Let $P(+ \| X) = 0.97$, $P(- \| no X) = 0.95$, $P(X) = 0.5$.

First look suggests that the test is going to be inferior as only 97% of the people with disease X are correctly diagnosed.

---

### Evaluation of Setting 2

$$P(X | +) = \frac{P(+ | X) * P(X)}{(P(+ | X) * P(X) + P(- | no X) * P(no X))} = \frac{0.97*0.5}{(0.97*0.5 + 0.05*0.5)} = 0.951$$

This is not good.

$$P(no X | -) = \frac{P(- | no X) * P(no X)}{(P(- | no X) * P(no X) + P(- | X) * P(X))} = \frac{0.95*0.5}{(0.95*0.5 + 0.03*0.5)} = 0.969$$

This is good, but not as good as setting 1.

---

### Closing Note

If the test is not 100% accurate (true for almost all tests), there will be a trade-off between $P(X \| +)$ and $P(no X \| -)$. The appropriate choice of sample depends on the severity of disease and severity of misclassification. This choice becomes even more important during randomization for statistical modeling and hypothesis testing.

Consider the disease X = HIV. Hence the misclassifications will be: a patient who is wrongly diagnosed as HIV +ive (predicted -ive), a patient who is wrongly diagnosed as HIV -ive (predicted +ive). A medical expert should be consulted to adjust the weights for these misclassifications (I’m not an expert in medicine or human behavior). Based on the weights the sample P(X) can be chosen appropriately.