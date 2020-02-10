# Week1

## Bayesian approach to statistics

### Different between Frequentist and Bayesian

Frequentist inference vs Baysian inference

- Frequentists treat the objective(객관적)
  - Parameter(theta) is fixed
  - Data is random
  - Want to find optimal point
  - Work only when the number of data points is much bigger than the number of parameters
  - Use maximum likelihood to train model
    - Try to find parameters theta that maximize the likelihood
    - $$\hat{\theta} = \underset{\theta}{argmax} P(X|\theta) $$
- Bayesians treat it as subjective(주관적)
  - Parameter(theta) is random
  - Data is fixed
  - Work for arbitrary number of letter points
  - e.g. in neural network, weights(parameters) are random. training set(data) is fixed
  - What Bayesians will try to do is they would try to compute the posterior, the probability of the parameters given the data
    - 결과로 원인을 찾는게 목표임
    - $$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$$
    - It will compute posterior distribution

### Classification of Bayesian

#### Training

$$
P(\theta | X_{tr}, y_{tr}) = \frac{P(y_{tr}|X_{tr}, \theta)P(\theta)}{P(y_{tr} | X_{tr})}
$$

- Training data와 training data로 얻은 결과로 parameter 추정

#### Prediction

$$
P(y_{ts}|X_{ts}, X_{tr},y_{tr}) = \int P(y_{ts}|X_{ts}, \theta)P(\theta | X_{tr}, y_{tr}) d\theta
$$

- A weighted average of output of our model ***for all possible values of parameters***
- tr : training
- ts : test
- training 결과과 training에 사용한 데이터, test용 데이터로 test 결과 추정
- Marginalisation principle 
  - $$P(B|C) = \underset{i}{\sum}P(A_i|C)P(B|A_i,C)$$

### On-line learning

- Bayesian methods are really good for online learning.

$$
P_k(\theta) = P(\theta|x_k) = \frac{P(x_k |\theta)P_{k-1}(\theta)}{P(x_k)}
$$

- $$P_k(\theta)$$ : New Prior
- $$P(\theta | x_k)$$ : Posterior
- $$P(x_k|\theta)$$ : likelihood
- $$P_{k-1}(\theta)$$ : Prior
- Can use new posterior as a prior to the next experiment.

## How to define a model

The most convenient way to do this is called the Bayesian Network.(Don't mix up with Bayesian neural network)

- Nodes : random variables
- Edge : direct impact

### Probabilistic model from BN

Model : Joint probability over all variables
$$
P(X_1, X_2, ... X_n) = \prod_{k=1}^n P(X_k | Pa(X_k))
$$

- Pa : Parents

