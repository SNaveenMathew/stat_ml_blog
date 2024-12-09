---
layout: post
date: 2019-08-20 12:00:00 -0500
image: /data/regressionfit.gif
---

## Equivalence of MLE and OLS in Linear Regression

### Introduction

A linear regression model can be described using $Y = \alpha + \beta X + \epsilon$. This simplifies to the following form on the observed data: $y=\alpha + \beta + e \tag{Equation 1}$. After obtaining the maximum likelihood estimate for the coefficients using the sample, this equation simplifies to $\hat{y} = \hat\alpha + \hat\beta x$. The objective of this (short) article is to use the assumptions to establish the equivalence of OLS and MLE solutions for linear regression.

### Important Model Assumptions

1. True underlying distribution of the errors is Gaussian
2. Expected value of the error term is 0 (known)
3. Variance of the error term is constant with respect to x
4. The 'lagged' errors are independent of each other

### Full Likelihood

For an observation e from a Gaussian distribution with 0 mean and constant variance, the likelihood is given by $L(e\|\alpha, \beta) = \frac{exp(-\frac{e^2}{2\sigma^2})}{\sqrt{2\pi\sigma^2}}$. Given the whole data set of n observations, assuming the residues are realizations of the iid (independent and identically distributed) Gaussian error, the likelihood can be written as:

$$L(\vec{e}|\alpha,\beta)=\prod_{i=1}^{n}\frac{exp(\frac{-e_i^2}{2\sigma^2})}{\sqrt{2\pi\sigma^2}}=\frac{exp(-\frac{\sum_{i=1}^{n}e_i^2}{2\sigma^2})}{(\sqrt{2\pi\sigma^2})^n}$$

Since log is a monotonous transformation, the maximum likelihood estimate does not change on log transformation:

$$l(\vec{e}|\alpha,\beta)=log(L(\vec{e}|\alpha,\beta))=-\frac{\sum_{i=1}^{n}e_i^2}{2\sigma^2}-\frac{n}{2}(log(2\pi) + 2log(\sigma))$$

Substituting the maximum likelihood estimate:

$$\hat\alpha, \hat\beta = argmax_{\alpha,\beta} l(\vec{e}|\alpha,\beta) = argmax_{\alpha,\beta}\bigg[-\frac{\sum_{i=1}{n}e_i^2}{2\sigma^2}\bigg] - \frac{n}{2}(log(2\pi) + 2log(\sigma))$$

Removing the constant terms:

$$\hat\alpha, \hat\beta = argmax_{\alpha,\beta} \sum_{i=1}^{n}-e_i^2$$

Substituting $e$ from equation 1, we get:

$$ \hat\alpha, \hat\beta = argmax_{\alpha,\beta} \sum_{i=1}^{n}-(y-\beta x -\alpha)^2 $$

Maximizing -z is equivalent to minimizing z, therefore $ \hat\alpha, \hat\beta = argmin_{\alpha,\beta} \sum_{i=1}^{n} (y-\beta x -\alpha)^2$

### All Assumptions

1. Relationship between independent variable and dependent variables is linear
2. True underlying distribution of the error is Gaussian with 0 mean
3. Independent variables do not exhibit high level of multicollinearity
4. No autocorrelation: 'lagged' error terms are independent
5. No heteroskedasticity (already used): variance of the error is independent of X and is constant throughout
6. Multivariate normality of independent variables (not required, but helpful) for proving few special properties
7. The independent variables are measured without random error. Therefore, X and x are not random

### Additional Resources

1. [Equivalence of ANOVA and linear regression](https://snaveenmathew.github.io/stat_ml_blog/2019/01/11/Simple-Linear-Regression-and-ANOVA.html)
2. [Simple physics for an intuitive understanding of linear regression](https://snaveenmathew.github.io/stat_ml_blog/2018/06/05/OLS-Linear-Regression-Hyperplane-of-zero-net-force-and-torque.html)