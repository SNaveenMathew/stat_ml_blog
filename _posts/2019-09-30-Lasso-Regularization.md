## Lasso Regularization on Linear Regression and Deep Learning Models

### Introduction

#### Curve fitting — under and over fitting

As discussed in my [previous article](https://snaveenmathew.github.io/stat_ml_blog/2018/07/06/A-Short-Note-on-Regularization.html), issues with 'curve fitting' occur when the problem is ill-posed. Underfitting is usually not a big problem because we have the option to expand the feature set by acquiring/engineering new features. However, overfitting is not easy to handle.

#### Best subset selection in regression

Consider a linear regression with $p$ predictor variables. Assume that the whole data set is used for training. It is known that the training set $R^2$ never decreases on addition of features. Therefore, $R^2$ is not always a good measure of goodness of fit. Adjusted-$R^2$, Mallow’s $C_p$, AIC, BIC, etc. are used to measure the goodness of fit. However, a priori knowledge does not exist on the change in values of these measures upon addition/removal of predictor variables. Therefore, all $2^p-1$ distinct models may be required to judge the 'best subset' of features required to model the outcome variable. However, this is computationally very expensive. This necessitates an appropriate way to reduce variables without building exponentially large number of models. Lasso penalty helps in achieving this goal partially.

### Formulation

#### Linear regression and normal equation

$$Y=X\beta + \epsilon \tag{Regression equation}$$

$$y=x\hat\beta^{(MLE)}+\hat{e}^{(MLE)} \tag{Linear regression estimated on a sample}$$

$$L(\beta)=||y-x\beta||_2^2 \implies \hat\beta^{(MLE)}=argmin_\beta ||y-x\beta||_2^2 \tag{OLS solution — same as MLE under certain conditions}$$

$$\frac{\partial L}{\partial\beta}|_{\hat{\beta}^{(MLE)}}=0 \implies x^T(y-x\hat\beta^{(MLE)})=0 \implies \hat\beta^{(MLE)}=(x^Tx)^{-1}x^Ty \tag{Normal equations}$$

We observe that the OLS solution does not exist (may not be unique) if the covariance matrix is non-invertible.

### Lasso formulation

$$Y=X\beta + \epsilon \tag{Regression equation}$$

$$y=x\hat\beta^{(Lasso)}+\hat{e}^{(Lasso)} \tag{Lasso solution estimated on sample}$$

$$L(\beta)=||y-x\beta||_2^2 + \lambda |\beta|_1 \implies \hat\beta^{(Lasso)}=argmin_\beta \big[||y-x\beta||_2^2 + \lambda |\beta|_1\big] \tag{Lasso solution}$$

Analytical solution does not exist for this minimization. Also gradient descent is not guaranteed to converge on this loss function even though it is convex. An alternate formulation is required to computationally solve this problem.

#### Solution

##### Soft thresholding for orthogonal covariance

$$L(\beta)=||y-x\hat\beta^{(OLS)}+x\hat\beta^{(OLS)}-x\beta||_2^2+\lambda|\beta|_1$$

$$L(\beta)=||y-x\hat\beta^{(OLS)}||_2^2 - 2(y-x\hat\beta^{(OLS)})^T x(\hat\beta^{(OLS)}-\beta) + ||x\hat\beta^{(OLS)}-x\beta||_2^2+\lambda|\beta|_1; y-x\hat\beta^{(OLS)} = \hat{e}^{(OLS)} \perp x$$

$$\implies \hat\beta^{(Lasso)}=argmin_\beta \big[X(\hat\beta^{(OLS)}-\beta)\big]^T \big[X(\hat\beta^{(OLS)}-\beta)\big] + \lambda |\beta|_1$$

$$\implies \hat\beta^{(Lasso)}=argmin_\beta (\hat\beta^{(OLS)}-\beta)^TX^T X(\hat\beta^{(OLS)}-\beta) + \lambda |\beta|_1$$

Assuming $X^T X=I$

$$\hat\beta^{(Lasso)}=argmin_\beta (\hat\beta^{(OLS)}-\beta)^T(\hat\beta^{(OLS)}-\beta) + \lambda |\beta|_1$$

$$\implies \hat\beta^{(Lasso)}=argmin_\beta (\hat\beta^{(OLS)}_i-\beta_i)^2 + \lambda |\beta_i|_1 + \sum_{j=1;j\neq i}^{p}\big[(\hat\beta^{(OLS)}_j-\beta_j)^2 +\lambda |\beta_j|_1\big]$$

$$\implies \hat\beta_i^{(Lasso)} = argmin_{\beta_i} \big[(\hat\beta^{(OLS)}_i-\beta_i)^2 +\lambda |\beta_i|_1\big] \tag{Separating out the dimensions}$$

$$\hat\beta_i^{(Lasso)}
\begin{cases}
    \hat\beta_i^{(OLS)}-\frac{\lambda}{2}& \text{if } \hat\beta_i^{(OLS)}>\frac{\lambda}{2}\\
    0              & \text{if }|\beta_i^{(OLS)}| \le \frac{\lambda}{2}\\
    \hat\beta_i^{(OLS)}+\frac{\lambda}{2}& \text{if } \hat\beta_i^{(OLS)}<-\frac{\lambda}{2}
\end{cases}
\tag{Soft thresholding}$$
