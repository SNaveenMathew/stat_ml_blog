---
layout: post
title: A Short Note on Regularization
date: 2018-07-06 12:00:00 -0500
image: ../../../data/regressionfit.gif
---

## A Short Note on Regularization

### Introduction

In deep learning era, regularization one of the most commonly used methods for reducing 'overfitting'. In this article I'd like to introduce regularization by building a step-by-step intuition. I assume that the readers are aware of basic terms used in machine learning (eg: training examples, predictors, training set, test set, overfitting, etc).

---

### Problem Setup, Asssumptions, etc.

Let us concentrate on building a linear model on training set. This learning problem is generally ill-posed. In other words, if y is the dependent variable and x is (are) the independent variable(s), $Ax = y$ does not have a unique solution (almost all practical problems fall in this category).

---

### Further What-If Analysis

There are 2 possible cases that are encountered practically:

1. Number of equations >> number of variables which can be approximated using ordinary least square (OLS) regression. It is expected to generalize (work well for the test set).
2. Number of variables >> number of equations which leads to multiple solutions. OLS regression fit will be perfect for the training set, but will yiel---d miserable results for the test set.

The second, which defines an over-determined system, can be considered as a primitive example of 'overfitting' (there are many other cases of overfitting): Imagine adding a testing row $(X^{(j)}, y^{(j)})$ which does not fit in the over-determined system of equations. Each feasible solution that was determined for the system of equations will give a different predicted value $\hat{y}$, some of which will differ drastically from the ground truth $y$. This defeats the purpose of predictive modeling.

---

### Overcoming Overfitting

There are few ways of overcoming this issue (described in layman terms):

1. Decrease the number of predictors
2. Decrease the importance given to the variables in the predictive model (regularization; this sometimes intersects with solution #1)
...

A special note on step 2: Decreasing the importance given to variables pulls the model towards the most basic model, that is the null model: $\hat{Y}=average(Y_{train})$ which discards all predictors. A more rigorous treatment of regularization involves use of Bayesian statistics. This will be discussed in another article.