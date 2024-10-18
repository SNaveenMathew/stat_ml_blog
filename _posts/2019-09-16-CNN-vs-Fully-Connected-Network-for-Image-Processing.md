## CNN vs Fully-Connected Network for Image Processing

### Introduction
The objective of this article is to provide a theoretical perspective to understand why (single layer) CNNs work better than fully-connected networks for image processing. Linear algebra (matrix multiplication, eigenvalues and/or PCA) and a property of sigmoid/tanh function will be used in an attempt to have a one-to-one (almost) comparison between a fully-connected network (logistic regression) and CNN. Finally, the tradeoff between filter size and the amount of information retained in the filtered image will be examined for the purpose of prediction. For simplicity, we will assume the following:

1. The fully-connected network does not have a hidden layer (logistic regression)
2. Original image was normalized to have pixel values between 0 and 1 or scaled to have mean = 0 and variance = 1
3. Sigmoid/tanh activation is used between input and convolved image, although the argument works for other non-linear activation functions such as ReLU. ReLU is avoided because it breaks the rigor of the analysis if the images are scaled (mean = 0, variance = 1) instead of normalized
4. Number of channels = depth of image = 1 for most of the article, model with higher number of channels will be discussed briefly
5. The problem involves a classification task. Therefore, C > 1
6. There are no non-linearities other than the activation and no non-differentiability (like pooling, strides other than 1, padding, etc.)
7. Negative log likelihood loss function is used to train both networks

#### Symbols and Notation

Symbols used are:

1. $x$: Matrix of 2-D input images
2. $W_1$, $b_1$: Weight matrix and bias term used for mapping
Raw image to output in fully-connected network
Filtered image to output in CNN
3. p: Output probability
4. $X_1$: Filtered image
5. $x_1$: Filtered-activated image

Two conventions to note about the notation are:

1. Dimensions are written between {}
2. Different dimensions are separated by x. Eg: {n x C} represents two dimensional 'array'

### Model definition

#### Fully-Connected Network

$$Z_{1[n \times C]}=x_{n\times(n_y*n_x)}WW_{1[(n_y*n_x)\times C]} + b_{1[1\times C]} \tag{FC1: pre-output layer}$$

$$p_{[n\times C]} = F_{softmax}(Z_1)=\frac{[e^Z_1]_{[n\times C]}}{[\sum_{c=1}^{C} e_{C}^{Z_i}]_{n\times 1}} \tag{FC2: estimated probability}$$

#### Convolution Neural Network

$$X_{1[n\times(y-k_y+1)\times(x-k_x+1)]}=x_{[n\times n_y \times n_x]}K_{[k_y\times k_x]} \tag{C1: filtered image}$$

$$x_{1[n\times (n_y-k_y+1) \times (n_x-k_x+1)]} = \sigma(X_1) \tag{C2: filtered-activated image}$$

$$\sigma^{(sigmoid)}(X_1)=\frac{e^{X_1}}{1+e^{X_1}}; \sigma^{(tanh)}(X_1)=\frac{e^{X_1}-e^{-X_1}}{e^{X_1}+e^{-X_1}} \tag{Activation functions}$$

$$Z_{1[n\times C]}=x_{1[n\times (n_y-k_y+1) \times (n_x-k_x+1)]} \odot W_{1[(n_y-k_y+1) \times (n_x-k_x+1) \times C]} + b_{1[1\times C]} \tag{C3: pre-output layer}$$

$$p_{n\times C} = F_{softmax}(Z_1) = \frac{e^{Z_1}}{1+e^{Z_1}} \tag{C4: estimated probability}$$

### The Mathematics

#### Reducing the CNN to a fully-connected network

Let us assume that the filter is square with $k_x = 1$ and $K(a, b) = 1$. Therefore, $XX_1 = x$. Now the advantage of normalizing x and a handy property of sigmoid/tanh will be used. It is discussed below:

#### Required property of sigmoid/tanh

