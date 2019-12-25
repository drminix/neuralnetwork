# Neural Networks and Deep Learning
## What is a neural network

- Deep learning refers "training very large neural network". 
- Each neuron performs certain operations like computing a weighted sum followed by a non-linear activation function operation such as ReLU.
- Neural network is formed by stacking multiple layer of neurons.
Example neural network is shown below.
![alt text](img/simplenn.png)
- Input layer and hidden layer are density connected.  Every input feature is connected to every unit in the hidden layer.
- Rectified Linear Unit (ReLU) function. Rectify means taking a max of 0.

## Supervised learning
- In supervised learning, the training data you feed to the algorithm includes the desired solutions, called *labels*.
- Typical task of supervised learning: (1) classification such as spam filers and (2) regression
- Examples: (1) Linear regression, (2) Logistic Regression, (3) Support Vector Machines, Decision Trees and Random forests, (5) Neural networks
- Applications of Neural Network in supervised learning: Online Advertising, Photo tagging, Speech recognition, Machine translation, Autonomous driving
![alt text](img/supervisedlearning.png)
- Examples of neural networks
![alt text](img/neuralnetworkexamples.png)
- Structured Data: databases of data. Each of the feature has very well defined meaning 
- Unstructued Data: Audio, Image, Text. 
- Most of the successful application of DL has been in supervised learning.

## Unsupervised learning
- In unsupervised learning, the training data is unlabeled. The system tries to learn without a teacher
- Examples: (1) Clustering, Dimension Reduction, and etc. 

## Why Deep learning is taking off?
- Scale drives deep learning process. "Scale" means both (1) labelled data & (2) size of NN.
- GPU + More Data(Digitization of data) + Algorithm advances in NN = Big Bang!
- One of the huge breakthrough in NN has been switching from SIGMOD function to ReLU function!. It's more computationally efficient!  Learning is slow in SIGMOD when gradient is nearly zero.
- Use of ReLU made the computation of gradient descent much faster.

## Logistic Regression as a Neural Network
- Logistic regression is a type of regression analysis method often used to predict the value of binary dependent variable.
- In Logistic regression, we use a logistic function to constrain the value of the dependent variable such that its value can't be smaller than 0 and bigger than 1.
- In Logistic regression, we use a logstic model to model the probability of a certain class or event existing. 
- Sigmoid function is a mathematical function having a characteristic "S"-shaped curve. Standard choice for a sigmoid function is the logicstic function whose formula is defined as S(x) = 1 / (1+e^(-x)). There are other sigmoid functinos such as arctangent function.

## Backpropagation and gradient descent.
- Training neural network consists of (1) forward propagation to get predictions (2) backpropagation to compute derivatives and (3) gradient descent update to adjust parameters.
- In forward propagation, we propagate our training set through the neural network to get our estimated labels. Then we can compute the cost function by subtracting the true labels with our estimated labels. By the way, we can't use a simply least square as a cost function since it is not convex;thus has a local minima. We need to use a convex function so that we can always reach a global minimum. 
- We use a gradient descent method to minimize our cost function. Since the cost function is a convex function, it is guaranteed to reach a global minimum with gradient descent method.

## Tips
- Always give an explicit size using reshape function. Avoid using a rank-1 array. Either use a row or column vector. You can enforce that by using `keepdims=True` parameter in numpy.
- reshape operation is O(1). 
- Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization.
- Normalization means changing x to x/||X||. That is dividing each row vector x by its norm
- For example:  `x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)`

## Common steps for pre-processing a new data set:
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
- To flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d,a). 
- `X_flatten = X.reshape(X.shape[0],-1).T`
- This is useful when a matrix represents (number_training_set,height,width,channel)
- This results in (flattened_image, number_training_set)
- "Standardize" the data. Subtract the mean from each example and divide it by the standard deviation. 
- It converges faster! For image data, we can just divide each pixel by the maximum value of a pixel channel

## Main steps for building a neural network 
1. Define the model structure (such as number of input features, number of layers, type of layers, and etc.)
2. Initialize the model's parameters
3. Loop
   -- Forward propagation: calculate the current prediction 
   -- Compute loss
   -- Backward propagation: calculate current gradient for evey neuron
   -- Gradient descent: Update parameters 

## Activtion functions
1. Sigmoid(logistic function): a = 1/(1+e^-z)
-- Never use it unless in the output layer if you are doing a binary classification since its output is constrained to be between 0 and 1
-- Computationally expensive.
-- Slow learning when z is large or small!!

2. Tanh function: a = (e^z-e^-z) /(e^z+e^z)
-- Superior than sigmoid!!

3. ReLU a=max(0,z)
-- Most commonly used
-- much faster learning!
-- Computationally inexpensive!!
-- Derivative is 0 when z is negative but z is usually greater than 0.
-- Derivative of z iss not defined when z is 0 but the chance of having exact 0 is very low. We can define the derivative to be either 0 or 1.

4. Leaky ReLU a=max(0.01z,z)
-- works better than ReLU but not used much

## Why do we need non-linear activation fns?
-- If you don't use non-linear activation fn and use linear activation fn, the neural network can be simplified to linear activation function and all the hidden layer becomes useless since a composite of linear hidden layer is just a linear fn. For example, deep neural network with a lot of hidden layers can be simplified to 1 layer network. This cannot capture any complex behaviors.
- We can use a linear activation fn in the output layer when the dependent variable is numeric-scale values.

## Random initialization
- Need to initialize W to random numbers to zeroes otherwise all neurons in the same layer will be doing the same computations. 
- By initializing w with random numbers, we can avoid symmetry problem. This is commonly called "symmetry breaking".
- By initializing w with random numbers, we can avoid symmetry problem. This is commonly called "symmetry breaking".
- Initialize to small number to avoid big or small z values which leads to small derivatives which in turn results in "slow learning".

## Intuition about deep representation. Why does it work so well?
1. Simple Intuition" input -> simple features -> complex features
2. From Circuit Theory: So this result comes from circuit theory of which pertains the thinking about what types of functions you can compute with different AND gates, OR gates, NOT gates, basically logic gates. So informally, their functions compute with a relatively small but deep neural network and by small I mean the number of hidden units is relatively small. But if you try to compute the same function with a shallow network, so if there aren't enough hidden layers, then you might require exponentially more hidden units to compute
3. Just Branding: Deep NN used to be called a NN with a lot of hidden layers

## Forwarding and backward function
NN consists of two different types of units:
(1) Forward function unit: It takes in a[l-1] and outputs a[l]. It uses parameters w[l] and b[l] and caches z[l]. These values are used duing backpropagation for computing the derivaties
(2) Backward function unit: It takes in da[l] and outputs da[l-1]. It uses the cached values from the forward function and computes w[l],b[l], and dz[l]. Each unit computes dw[l] and b[l] which are then used to update w[l] and b[l] parameters for given layer l.

## NOTE - a lot of complexity comes from data not the algorithm. That is your NN can do unexpectedly complex task.

## Parameters and HyperParameters
Parameters: W[l] and b[l]
Hyper parameters: Which controls the paramters W[l] and b[l]
1. learning rate alpha
2. number of iteration
3. hidden layer
4. hidden units
5. choice of activation fn
6. Momentum
7. Minibatch size
8. Regularization parameters.

## What dos NN have to do with the brain?
1. Analogy of the neural network to the human brain is an over-simplification.
2. People started using human brain analogy which is a overly simplified analogy. It's hard to convey the intuition behind NN.
3. Neuroscientist have no idea what a single neuron is doing...

## Unbiased vs Consistent estimator
Estimator is said to be unbiased if the expected value of estimator is equal to the true parameter. That is E(X) = u. Estimator might be biased but consistent. In this case, estimator becomes unbiased estimator if we collect more datasets. 

## How to fix NN when it has high bias.
When our model has high bias, it means that our estimator is unbiased estimators, meaning that there is a difference between the expected value of our estimator, E(X) and the true parameter u. That is, bias = E(X) - u. In NN, it usually means that our NN model is not trainined enough with the training set. High bias can be alleviated by using (1) a bigger network (more neurons or deeper layers), (2) increasing the number of iterations, (3) or using a different type of NN.

## How to fix NN when it has high variance
When the prediction error for test set is high, it is most likely to have high variance problem. High Variance can be alleviated by using (1) more training dataset eith by collecting more data set or using data augmentation, (2) using L-2 regualrization technique to penalize the cost function if W gets too large, (3) using a technique like dropout, (4) early stopping to stop training when the prediction errors on training set and dev set start to diverge. L-2 regularizaiton technique is also called weight decay, since the regularizaiton term forces W to become smaller at every iteration.


## Optimization
Deep learning is a highly iterative process so it is important to be able to try out ideas quickly. It is especially important when the training set is very large. So how do we make NN train faster?

1. normalize input feature vector X so that every element is in the same scale.
Standardlize them by $x-mean /variance$. 

2. Pervent weight vanishing or exploding by using clever weight initialization method like xaiver random initilization method.

3. Batch gradient descent vs Mini-batch gradient descent
In batch gradient descent, you go through every examples in the training-set before you update the training set. The downside of gradient descent is that it takes long before we update the network. In mini-batch gradient descent, we don't wait until you process all the training dataset. Training set is divided into a smaller subset and gradient descent update is performed for each subset. If we set the mini batch size to be 1, then we are doing a stochastic gradient descent update. The biggest problem with stochastic gradient descent is that it is slow since we are not taking advantage of vectorization.

How do we choose the right mini-batch size? If the training set is small, use a batch gradient descent(size = m). If the training set is large, use a mini-batch gradient descent(size = 2^6(64), 2^7(128), 2^8(256)) Make sure that it fits CPU/GPU memory.

4.  Gradient descent with momentum
It almost always converges faster than standard gradient descent algorithm. Basic idea is to compute an exponentially weighted average of gradients and then use that gradient to update your weight. Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent. Instead of using $dw$ and $db$ to update your parameters, use exponentially weighted moving averages. that is use $V_{dw}$ and $V_{db}$ where $V_{dw} = \beta V_{dw} + (1-\beta) dw$. $W = W - alpha * V_{dw}$ It smooth out the  steps of gradient descent. There is new hyper parameter, $\beta = 0.9$

- Exponentially weighted moving average has a formula of $V_t = \beta V_{t-1} + (1-\beta) V_{t-1}$. $V_t$ is approximately averaging over $1/(1-\beta) $ days. For example, $\beta=0.9$ represents 10 days. Weight of each day varies with the formula $(0.9)^x$. Not as accurate as moving window but it has very memory foot print since we only need to store one variable, $\theta_t$

- $V_t = \beta V_{t-1} + (1-\beta) V_{t-1}$ is not accurate during the initial phase since we start with $V_0=0$. One way to fix this issue is to use the following bias correction method. So instead of taking $V_t$, take $V_t/(1-\beta^t)$. When t is large, it becomes $V_t$ so this fomrula becomes identical to the original exponentially weighted moving average formula. It is not used much in practice.

5. RMSprop
- RMSprop stands for root mean square prop. It has similar effect as momentum. It has effect of damping out the oscillation. It can also speed up gradient descent.
- Basically we divide the parameters such as $dw$ by its root mean squared so that if the target parameter is large, $S_{dw}$ will be big thus dividing $dw$ with $S_{dw}$ will slow down the update.

- $S_{dw} = \beta S_{dw} + (1-\beta) dw^2$
- $W = W - alpha * (dw / \sqrt(S_{db}+e))$


6. ADAM(Adaptive Moment Estimation) optimization algorithm
- Combination of momentum + RMSProp to dampen out the oscillation! It converges faster than the standard gradient descent.
- Commonly used algirhtm proven to be very effective. It has the following parameters. learning rate(a), $\beta_1$ for momentum typially set to 0.9, $\beta_2$ for RMS prop typically set to 0.999, and the epsilon typically set to $10^{-8}$
- Typically, values of $\beta_1$ and $\beta_2$ are fixed and we only need to tune \alpha.

7. Hyper parameter tuning -- learning rate decay
- Slowly reduce the learning rate as it converges.
- method1: $alpha = 1/(decay_rate * epoch_number) * alpha_0$
- there are other method such as exponential decay like $alpha = 0.95^{epoch_number}$

8. Problem of local optima
- In high-dimensional space, it's very highly unlike that you get stuck in local optima in all direction. It could be a saddle point (it can go down in other direction).
- Problem is with plateaus. Plateaus can really slow down learning and a plateau is a region where the derivative is close to zero for a long time. 

## Useful references
1. http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
2. https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
3. http://cs231n.github.io/neural-networks-case-study/
4. http://scs.ryerson.ca/~aharley/neural-networks/
