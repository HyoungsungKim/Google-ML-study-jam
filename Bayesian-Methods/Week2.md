# Week2

## Latent Variable Models

###  Latent Variable Model

What is latent variable?

- Latent variable is just a random variable which is unobservable to you nor in training nor in test phase.

#### Example

When random variables X are given
$$
p(x_1, x_2, x_3, x_4, x_5) = \frac{exp(-w^Tx)}{Z}
$$

- But we cannot calculate normalize constant Z. Because we don't know relation of each random variable
- To detour this problem, use latent variable `I`

$$
\begin{align}
p(x_1, x_2, x_3, x_4, x_5) & = \sum^{k}_{I=1}p(x_1, x_2, x_3, x_4, x_5 | I)p(I) \\
& = \sum^{k}_{I=1}p(x_1|I)...p(x_5|I)p(I)
\end{align}
$$

Pros of Latent Variable

- It can reduce the number of phases we have. And as a consequence of that, we can reduce the number of parameters. 
- Some other positive feature of latent variables, is that they are sometimes interpretable. 
  - This variable can be interpretable and you can compare using latent variables according to this scale of different people in your data set.

Cons

- Some downside of latent variable models is that they can be harder to work with. So, training latent variable model, you have to rely on a lot math. 

### Gaussian Mixture Model

- 클러스터 그룹을 하나의 가우시안 모델로 표현 할 수가 없음( low flexibility)
- Flexibility를 위해 N개의 가우시간 사용

$$
p(x|\theta) = \pi_1 \mathcal{N} (x | \mu_1 , \sum{_1}) + \pi_2 \mathcal{N} (x | \mu_2 , \sum{_2}) + \pi_3 \mathcal{N} (x | \mu_3 , \sum {_3})
$$

- location : $$\mu$$
- shape : covariance($$\sum{_N}$$)
- This model is sometimes called Gaussian Mixture Model, or GMM for short.
- It has better flexibility than gaussian, however it has more parameter to calculate

#### How can we fit it? 

How can we find its parameters? So it's pi, mu and sigma vectors and matrices. Well the simplest way to fit a probability distribution is to use maximum likelihood estimation.

- We want to find the values of parameters which maximize the likelihood, which is the density of our data set given the parameters. 
- And we want to maximize this thing with respect to the parameters. 

As usually in machine learning, ***we will assume that the data set consists of N data points, which are independent given our parameters***. Which basically means that we can factorize the likelihood. So the likelihood equals to the product of likelihoods of individual objects. 
$$
\max_{\theta} \prod^N_{i=1} p(x_i | \theta) = \prod^N_{i=1}(\pi_1 \mathcal{N}(x_i|\mu_1, \sum{_1}) + ...) \\
\begin{align}
\text{subject to } & \pi_1 + \pi_2 + \pi_3 = 1; \pi_k \ge 0; k = 1, 2, 3\\
\sum{_k} \succ 0;
\end{align}
$$

- The covariance matrices sigma cannot be arbitrary. Imagine that your optimization algorithm propose to use covariance matrix with all zeros. It just doesn't work. It doesn't define a proper Gaussian distribution. 
  - Because in the Gaussian distribution definition you have to invert this matrix, and you have to compute its determinant, and divide by it. So if you have a matrix which is all 0s, you will have lots of problems like division by 0, and stuff like that. 
  - So it's not a good idea to assume that the covariance matrix can be anything. 
- The important part is that it's a really hard constraint to follow, so it's hard to adapt your favorite stochastic gradient descent algorithm to always follow this constraint. So to maintain this property that the matrices are always positive semi-definite
- If you say that all the covariance matrices are diagonal, which means that the ellipsoids that correspond to the Gaussians cannot be rotated. They have to be aligned with the axes. In this case it's much easier to maintain this constraint. And you can actually use some stochastic optimization here. 

#### Summary

So to summarize, we may have two reasons to not to use this stochastic gradient descent here. 

- First of all, it may be hard to follow some constraints which you may care about, like positive semi-definite covariance matrices. 
- Second of all, expectation maximization algorithm, which can exploit the structure of your problem, sometimes is much faster and more efficient. 

So as a general summary: we discussed that the Gaussian mixture model is a flexible probability distribution, which can solve the clustering problem for you if you fit your data into this Gaussian mixture model. And sometimes it's hard to optimize with stochastic gradient descent, but there is this alternative which we'll talk about in the next section. 

### Training GMM

If we decided to use this kind of latent variable model, then it is reasonable to assume that latent variable `t` has prior distribution $$\pi$$, so it's exactly the weights of our Gaussians.
$$
p(t=c | \theta) = \pi_c
$$

- The latent variable `t` equals to some cluster numbersoft

$$
p(x|t=c, \theta) = \mathcal{N}(x | \pi_c, \sum{_c})
$$

Using latent variables
$$
\begin{align}
p(x|\theta) & = \pi_1 \mathcal{N} (x | \mu_1 , \sum{_1}) + \pi_2 \mathcal{N} (x | \mu_2 , \sum{_2}) + \pi_3 \mathcal{N} (x | \mu_3 , \sum {_3}) \\
& = \sum^3_{c=1}p(x|t=c, \theta)p(t=c|\theta)
\end{align}
$$
***If we know the sources, if we know the values of the latent variables, it's easy to estimate the parameters data***.

- Actually, if we don't have this hard assignments, but rather have soft assignments so some posterior distribution on `t`, which means that for each data point.
  - soft assignments : source를 모를 때?
- We don't assume that it belongs to just one cluster, but rather we assume that it belongs to all clusters simultaneously, all Gaussian simultaneously.
- What will some different probabilities being posterior P of `t`, given X and parameters. If we have these probabilities, it is also easy to estimate the parameters.

If we know the sources, no matter soft segments or hard segments, then we can easily estimate the parameters of our Gaussian mixture model. But on practice, we don't, right? We don't know the sources, so how can we estimate the sources?

- It turns out that if we know the parameters, so the Gaussians, their locations(mean) and their variances, then we can easily estimate the sources because we can use just the Bayes' rule to do it.
  - 가우시안의 평균과 분산을 알면 parameter(theta)를 이용해서 source(x)를 추정 할 수 있음
  - Given : $$p(x|t=1, \theta) = \mathcal{N}(-2, 1)$$
  - Find : $$p(t=1 | x, \theta)$$

#### Summary

- Gaussian Mixture Model is a flexible probabilistic model which you can fit into your data, and it allows you to solve the clustering problem, and also gives you a probabilistic model of your data. So you can for example sample from this model new data points.
- Expectation Maximization algorithm is something to train this Gaussian Mixture Model, and other latent variables models, as we will see in the next few videos. 
- But for now it's just for Gaussian Mixture Model. And ***sometimes it can be faster than Stochastic Gradient Descent***, and it also helps you to handle complicated constraints like making the covariance matrix positive semi definite on each iteration. 
- And expectation maximization suffers from local maximum. But you know it's kind of expected because the overall problem was NP hard. so the optimal solution in a reasonable time is not possible. 

## Expectation Maximization algorithm

General form of expectation maximization, which will allow you to train almost any latent variable model 

### Jensen's inequality and Kullback Leibler divergence

#### Jensen's inequality

In concave
$$
f(E[X]) \ge E[f(x)] \text{; f(x) is concave}
$$
In convex
$$
f(E[X]) \le E[f(x)] \text{; f(x) is convec}
$$

#### Kullback Leibler divegence

Kullback-Leibler divergence, which is a way to measure difference between two probabilistic distributions.

Kullback Leibler divegence
$$
\mathcal{KL}(q||p) = \int q(x) \log{\frac{q(x)}{p(x)}}dx
$$

- q(x)의 엔트로피와 q(x)를 이용해서 계산한 p(x)의 엔트로피를 비교함
  - $$q(x)\log{\frac{1}{p(x)}-q(x)\log{\frac{1}{q(x)}}}$$
- 만약 두 엔트로피의 차이가 0이라면(같다면), p(x)와 q(x)의 엔트로피는 같음.

#### Properties of KL divergence

1. $$\mathcal{KL}(q||p) = \mathcal{KL}(p||q)$$
2. $$\mathcal{KL}(q||q) = 0$$
3. $$\mathcal{KL}(q||p) \ge 0$$

### Expectation-Maximization algorithm

Our problem here is to maximize the likelihood of our data set. The density of the data set, the parameters with respect to parameters. And this is marginal likelihood because we don't have zero latent variables, so we have to marginalize the amount. 

#### General form of Expectation Maximization

$$
\begin{align}
\max_\theta (\log{p(X|\theta)}) & = \log \prod^N_{i=1} p(x_i | \theta) \\
& = \sum^N_{i=1} \log{p(x_i|p)}
\end{align}
$$

- One more step, we can substitute the marginal likelihood of the data object $$x_i$$ by its definition, which is sum of the joint distribution respect to the values of $$t_i$$
  - Use latent variable
  - Guess there are 3 clusters

$$
\log{p(X|\theta)} = \sum^N_{i=1}log \sum^3_{c=1} p(x_i, t_i = c | \theta)
$$

- It can be in principle maximized with stochastic gradient descent. But sometimes it's not the optimal choice to do.

Let's build a lower bound for this thing by using the Jensen's Inequality. Because last one cannot give optimal choice.
$$
\log{p(X|\theta)} = \sum^N_{i=1}log \sum^3_{c=1} p(x_i, t_i = c | \theta) \ge \mathcal{L}(\theta)
$$

- In this case, instead of maximizing this original margin of likelihood, ***we can maximize its lower bound instead***.
- But using just one lower bound is not enough(not helpful to find optimal)

$$
\log{p(X|\theta)} = \sum^N_{i=1}log \sum^3_{c=1} \frac{q(t_i=c)}{q(t_i=c)} p(x_i, t_i = c | \theta)
$$

- It doesn't change anything, right, because we just like multiplied this thing by 1. But now we can treat this `q` as weights in the Jensen's inequality. So we can treat this `q` as alphas, and we can treat the joint distribution v of x and t, joined by q, as the points as v for Jensen's inequaliz. 
- Jensen's inequality.
  - $$q(t_i=c) \rightarrow \alpha_c$$
  - $$\frac{p(x_i,t_i=c|\theta)}{q(t_i=c)} \rightarrow v_c$$

$$
\log(\sum_c \alpha_c v_c) \ge \sum_c \alpha_c log(v_c)
$$

Therefore,
$$
\sum^N_{i=1}log \sum^3_{c=1} \frac{q(t_i=c)}{q(t_i=c)} p(x_i, t_i = c | \theta) \ge \sum^N_{i=1}\sum^N_{c=1}q(t_i = c)\log{\frac{p(x_i, t_i=c|\theta)}{q(t_i=c)}}
$$

#### Summary

First of all, we build a lower bound on the local likelihood
$$
\log{p(X|\theta)} \ge \mathcal{L}(\theta, q) \text{ for any q}
$$

- q : Variational parameters

E-step
$$
q^{k+1} = \underset{q}{argmax} \mathcal{L}(\theta^k, q)
$$

- On the E-step, fix theta and maximize with respect to q, maximize the lower bound with respect to q. 

Q-step
$$
\theta^{k+1} = \underset{\theta}{argmax} \mathcal{L}(\theta, q^{k+1})
$$

- M-step, fix q and maximize the lower bound with respect of theta.

### E-step detail

Gap : between logP and inference
$$
GAP = \log{p(X|\theta) - \mathcal{L{(\theta,q)}}} = \sum^N_{i=1}KL(q(t_i)||p(t_i|x_i, \theta))
$$

- Maximize $$\mathcal{L}$$ respect to q
  - It means minimize GAP

$$
q^{k+1} = \underset{q}{argmin}\mathcal{KL}[q(T)||p(T|X,\theta^k)]
$$

### M-step details

$$
\theta^{k+1} = \underset{\theta}{argmax}\mathbb{E}_{q^{k+1}}[\log{p(X,T|\theta)}]
$$

#### Convergence guaranties

$$
\log{p(X|\theta^{k+1})} \ge \mathcal{L}(\theta^{k+1}, q^{k+1}) \ge \mathcal{L}(\theta^k, q^{k+1}) = \log{p(X|\theta^k)}
$$

- On each iteration EM doesn't decrease the objective
- Guarantied to convergence to local maximum(or saddle point)

### Summary

- Method for training Latent Variable Models
- Handles missing data
- Sequence of simple task instead on one hard
- guaranties to converge
- Helps with complicated parameter constraints 
- Numerous extensions:
  - Variational E-step: restrict the set of possible q
  - Sampling on M-step

We can treat missing values as latent variables and still estimate the Gaussian parameters. But with one-dimensional data, if a data point has missing values, it means that we don’t know anything about it (its only dimension is missing) and not even the smartest latent variable model can extract information from a point like this.

- So the only thing that is left is to throw away points with missing data.
- Note that we also don’t need EM to estimate the mean vector (i.e. we need it only for the covariance matrix) in the multi-dimensional case: since each coordinate of the mean vector can be treated independently, we can treat each coordinate as one-dimensional case and just throw away missing values.