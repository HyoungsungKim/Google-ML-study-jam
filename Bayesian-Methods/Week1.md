# PWeek1

## Bayesian approach to statistics

### Different between Frequentist and Bayesian

Frequentist inference vs Baysian inference

- Frequentists treat the objective(객관적)
  - ***Parameter(theta) is fixed***
  - ***Data is random***
  - Parameter가 고정되어 있기 때문에 data(결과)는 해봐야지 알 수 있다고 생각하는 것
  - Work only when the number of data points is much bigger than the number of parameters
  - Use maximum likelihood to train model
    - ***Try to find parameters*** theta that maximize the likelihood
    - Parameter가 고정되어 있기 때문에 확률 추정에 parameter 사용
    - $$\hat{\theta} = \underset{\theta}{argmax} P(X|\theta) $$
- Bayesian treat it as subjective(주관적)
  - ***Parameter(theta) is random***
  - ***Data is fixed***
  - Parameter를 안다면 data를 추정 할 수 있음
  - Work for arbitrary number of letter points
  - e.g. in neural network, weights(parameters) are random. training set(data) is fixed
  - What Bayesian will try to do is they would try to compute the posterior, the probability of the parameters given the data
    - Data가 고정되어 있기 때문에 확률 추정에 data 사용
    - $$P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}$$
    - It will compute posterior distribution
    - $$P(\theta|X)$$ : posterior
    - $$P(X|\theta)$$ : likelihood
    - $$P(X)$$ : evidence
    - $$P(X)$$ : prior (or regularizer)

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

- Bayesian methods are really good for on-line learning.
- On-line learning : 현재 값을 그 다음 값을 얻기 위해 바로 사용 함.

$$
P_k(\theta) = P(\theta|x_k) = \frac{P(x_k |\theta)P_{k-1}(\theta)}{P(x_k)}
$$

- $$P_k(\theta)$$ : New Prior
- $$P(\theta | x_k)$$ : Posterior
- $$P(x_k|\theta)$$ : likelihood
- $$P_{k-1}(\theta)$$ : Prior
- Can use new posterior as a prior to the next experiment.

### How to define a model

The most convenient way to do this is called the Bayesian Network.(Don't mix up with Bayesian neural network)

- Nodes : random variables
- Edge : direct impact

### Probabilistic model from BN

Model : Joint probability over all variables
$$
P(X_1, X_2, ... X_n) = \prod_{k=1}^n P(X_k | Pa(X_k))
$$

- Pa : Parents

## Conjugate prior

Conjugate : prior와 posterior가 같인 분포이면 conjugate하다고 함

### Analytical inference

$$
P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}
$$

- If we know $P(X)$, then we would able to make new X.

### Maximum a posterior (MAP)

$$
\begin{align}
\theta_{MP} & = \underset{\theta}{argmax}P(\theta|X) \\
& = \underset{\theta}{argmax}\frac{P(X|\theta)P(\theta)}{P(X)} \\
& = \underset{\theta}{argmax} P(X|\theta)P(\theta)
\end{align}
$$

- We can get a Maximum a posterior without evidence(P(X))
- However, it has a lot of problems
  - Major one is that it is not invariant to re-parameterization
    - For example, MAP of gaussian is changed after sigmoid is applied
  - Next problem is we cannot use as a prior
    - Shortly, we cannot a information for next prior from current prior
    - On-line learning을 할 때 더 많은 정보를 얻을 수가 없음

$$
\begin{align}
P_k(\theta) &= P(\theta|X_k) \\
&= \frac{P(X_k|\theta)P_{k-1}(\theta)}{P(X_k)} \\
&= \frac{P(X_k|\theta)\delta(\theta - \theta_{MP})}{P(X_k)} \\
&= \delta(\theta - \theta_{MP})

\end{align}
$$

- Prior도 $$\theta$$가 $$\theta_{MP}$$ 일때 최대 이기 때문에 추가정인 정보가 전파 되지 않음

#### Summary

Pros

- Easy to compute

Cons

- Not invariant to re-parameterization
- cannot use as a prior
- Finds untypical point
- Cannot compute credible intervals
  - For example, when we got a 100 as a result, we cannot know it is $$100 \pm 0.001$$  or $$100 \pm 10$$

