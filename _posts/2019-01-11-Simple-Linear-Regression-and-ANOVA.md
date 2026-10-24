---
layout: post
date: 2019-01-11 12:00:00 -0500
image: /data/regressionfit.gif
---

## Simple Linear Regression and ANOVA

### Introduction

Linear regression analysis one of the earliest models used in pattern recognition and is one of the most commonly used algorithms in statistics. The goal of this article is to assist in interpretation of results of linear regression, rather than concentrating on technicalities of regression analysis and ANOVA.

### Background: Linear Regression

The purpose of linear regression is to study a variable $Y$ as a function of a variable $X$. In simple linear regression there is only 1 independent variable. It is called 'linear' because we represent the estimate of dependent variable $Y$ as a linear function of the independent variable $X$. The general form is: $Y = \beta_0 + \beta_1 X + \epsilon$.

A regression analysis of the form $Y = \beta_0 + \beta_1 X^2 + \epsilon$ is also linear in $X^2$. The coefficients are estimated by minimizing the sum of the squares of residues as show in [this](https://snaveenmathew.github.io/stat_ml_blog/2018/06/05/OLS-Linear-Regression-Hyperplane-of-zero-net-force-and-torque.html) article. The formulation is also called 'ordinary least square' (OLS). There are several restrictions to be satisfied to obtain a well behaved unique solution for the OLS problem. For simplicity, let us assume that the errors are 'well behaved' and the solution converges to a unique point, etc.

### Background: ANOVA

Analysis of variance is used to study the difference between means of groups. Since variations are observed within each group and also between different groups, studying variance and covariance also falls in the scope of ANOVA. In general, the terms that are encountered are within sum of squares (error), between sum of squares (regression model) and total sum of squares (null/no model). Dividing these terms by the corresponding degrees of freedom yields the corresponding mean square terms. Since the error variance is estimated, ratio of mean squares is taken to obtain a $F$-statistic.

### Simple Linear Regression and ANOVA

A not-so-obvious fact is that simple (ordinary least square) linear regression under standard conditions is a special case of ANOVA. General form of ANOVA can be written as $Y_i - \bar{Y} = Y_i - \hat{Y_i} + \hat{Y_i} - \bar{Y}$.

$$\sum(Y_i - \bar{Y})^2 = \sum(Y_i - \hat{Y_i})^2 + \sum(\hat{Y_i} - \bar{Y})^2 + 2\sum(Y_i - \hat{Y_i})(\hat{Y_i} - \bar{Y}) \tag{1}$$

$$\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1} X \tag{2}$$

Substituting equation $2$ in equation $1$, we get:

$$\sum(Y_i - \bar{Y})^2 = \sum(Y_i - \hat{Y_i})^2 + \sum(\hat{Y_i} - \bar{Y})^2 + 2\sum e_i(\hat{\beta_0} + \hat{\beta_1} X - \bar{Y}) = \sum(Y_i - \hat{Y_i})^2 + \sum(\hat{Y_i} - \bar{Y})^2$$

Notice that the property from equations $1$ and $2$ of [this](https://snaveenmathew.github.io/stat_ml_blog/2018/06/05/OLS-Linear-Regression-Hyperplane-of-zero-net-force-and-torque.html) article on OLS solution was used in the derivation. Effectively, the total sum of squares has been decomposed into sum of squares of two components. We obtain the mean squares by dividing these numbers by the corresponding degrees of freedom. This is the relationship between simple linear regression and ANOVA — recall two factor ANOVA.

### Model Significance

The variance of the error is unknown and is estimated as $\hat{\sigma}=\frac{RSS}{df_{residue}}$. Therefore, $\chi^2$ test cannot be applied to obtain the statistical significance of the model. Statistical significance is established using $F$-test. $F$-test examines the evidence against the null hypothesis by comparing the sample F statistic with the critical value. Sample F statistic is given by: $F = \frac{(RSS_{NH}-RSS_{AH})/(df_{NH}-df_{AH})}{RSS_{AH}/df_{AH}}$.

For a simple linear regression model, we use the following hypotheses: $H_0: \beta_1=0$ vs. $H_1: \beta_1 \neq 0$. Notice that for a model with $p$ variables and $n$ data points $TSS=RSS_{NH}$, $df_{NH}=n-1$, $TSS-RSS_{AH}=Explained\ sum\ of\ squares$, $df_{AH}=n-p-1$. The test can be rewritten as:

$$F=\frac{Model\ explained\ sum\ of\ squares/df_{model}}{TSS(no\ model)/df_{no\ model}}; F_{crit}=F_{1-\alpha;p;n-p-1}$$

### Closing Note: Multiple Linear Regression

Let us examine the F test for multiple regression $F = \frac{(RSS_{NH}-RSS_{AH})/(df_{NH}-df_{AH})}{RSS_{AH}/df_{AH}}$. Let us assume that the full model has $p$ variables with coefficients $\beta_0^{(f)}, \beta_1^{(f)}, ..., \beta_p^{(f)}$. Let us assume that a reduced model was identified with $q$ with coefficients $\beta_0^{(r)}, \beta_1^{(r)}, ..., \beta_q^{(r)}$. ANOVA can be used to compare whether the reduced model is sufficient to explain the variance in outcome variable.

$$Y=\beta_0^{(f)} + \beta_1^{(f)}X_1 + ... + \beta_p^{(f)}X_p + \epsilon^{(f)} \tag{Model 1}$$

$$Y=\beta_0^{(r)} + \beta_1^{(r)}X_1 + ... + \beta_q^{(r)}X_q + \epsilon^{(r)} \tag{Model 2}$$

$F$-test can be used to test the hypothesis $H_0: \beta_{q+1}=\beta_{q+2}=...=\beta{p}=0$ vs. $H_1: At\ least\ one\ of\ \beta_{q+1}, \beta_{q+2}, ..., \beta{p}\ is\ non\ zero$.

$$F=\frac{(RSS_{model\ 1}-RSS_{model\ 2})/(p-q)}{RSS_{model\ 2}/df_{model\ 2}}; F_{crit}=F_{1-\alpha,p-q,n-p-1}$$

As a result, $F$-test also acts as a model/variable selection criteria. However, we made strong assumptions (such as homoskedasticity, no multicollinearity, etc.) which should be tested. [This](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/) article provides few details.