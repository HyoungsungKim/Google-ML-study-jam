# Poisson, Exponential and Gamma distribution

## Poisson

https://towardsdatascience.com/poisson-distribution-intuition-and-derivation-1059aeab90d

### Why did Poisson invent Poisson distribution?

To predict the # of events occurring in the future.

- More formally, to predict the probability of a given number of events occurring in a fixed interval of time.
- 고정된 시간 간격 안에서 사건이 발생하는 횟수에 대한 확률을 예측 하기 위해서

What is Poisson for? What are the things that only Poisson can do, but Binomial can't?

### The Shortcomings of the Binomial Distribution

***The problem with binomial is that it CANNOT contain more than 1 event in the unit of time***

- 1번의 시도에 1번의 사건만 일어나는 경우만 고려 할 수 있음
- The unit of time can only have 0 or 1 event.

>  The idea is, we can make the Binomial random variable **handle multiple events by dividing a unit time into smaller units**. By using smaller divisions, we can make the original unit time contain more than one event.

- Derive of Poisson distribution
  - In the Binomial distribution, the $ of trials(n) should be known beforehand.
  - The Poisson distribution, on the other hand, doesn't require you to know `n` or `p`
  - We are assuming `n` is infinity large and `p` is infinitesimal
    - $$n \rightarrow \infin$$
    - $$p \rightarrow 0$$
  - Using the limit, the unit times are now infinitesimal. We no longer have to worry about more than one event occurring within the same unit time. And this is how we derive Poisson distribution.
- ***The only what we need in Poisson is rate($$\lambda$$)***
- 포아송 분포에서는 발생 시행 횟수(n)나 확률(p)에 대해서 알 필요 없이, rate만 알면 됨.
- 실제 사는 세상에서는 rate만 아는 경우가 많음

## Exponential

$$P(T > t) = e^{-\lambda t}$$

https://towardsdatascience.com/what-is-exponential-distribution-7bdd08590e2a

### Why did we have to invent Exponential Distribution

To predict the amount of waiting time until the next event(i.e.,  success, failure, arrival, etc)

- 다음 사건 발생까지 대기 시간을 알기 위해 사용 함

For example

- The amount of time until the customer finishes browsing and actually purchases something in your store(success)
- The amount of time until the hardware on AWS EC2 fails (failures)
- The amount of time you need to wait until the bus arrives (arrival)

PDF of exponential : $$\lambda * e^{-\lambda t}$$

### X ~ Exp($$\lambda$$), Is the exponential parameter $$\lambda$$ the same as $$\lambda$$ in Poisson?

- $$\lambda$$ is not a time duration, but it is an event rate, which is the same as the parameter $$\lambda$$ in a Poisson process.

The confusion starts when you see the term *“***decay parameter**”, or even worse, the term “**decay rate**”, which is frequently used in exponential distribution. The *decay parameter* is expressed in terms of **time** (e.g., every 10 mins, every 7 years, etc.), which is **a** **reciprocal (1/λ) of the rate (λ) in Poisson.** Think about it: If you get 3 customers per hour, it means you get one customer every 1/3 hour.

- rate는 단위가 횟수
- 1/rate는 단위가 시간

>  Confusion-proof : Exponential‘s parameter λ is the same as that of Poisson process (λ).

### Derive the PDF of Exponential 

The definition of exponential distribution is the ***probability distribution of the time between the events in a Poisson process***.

If you want to model the probability distribution of “nothing happens during the time duration t,” not just during one unit time, how will you do that?

### Memoryless Property

Definition:

- $$P(T > a + b | T > a) = P(T > b)$$

For example, if the device has lasted nine years already, then memoryless means the probability that it will last another three years  (so, a total of 12 years) is exactly the same as that of a brand-new machine lasting for the next three years.

- $$P(T > 12 | T > 9) = P(T > 3)$$
-  To model this property— **increasing hazard rate —** we can use, for example, a [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution).

Then, when is it appropriate to use exponential distribution?

- Car accidents. It doesn’t increase or decrease your chance of a car accident if no one has hit you in the past five hours. This is why **λ is often called a hazard rate.**
- 한 도로에서 자동차 사고가 9번 발생했고, 3회 더 발생 할 확률은 3번 발생 할 확률과 같음

***Exponential distribution is the only continuous distribution that is memoryless(In discrete, geometric only)***

### Recap: Relationship between a Poisson and an Exponential distribution

- If the number of events per unit time follows a Poisson distribution, then the amount of time between events follows the exponential distribution.

포아송 분포 : 특정 횟수 시도 했을 때 사건이 얼마나 발생 했는지 보고 싶을 때?(Unit time단위로)

지수 분포 : 특정 횟수까지 사건이 발생 하지 않고 특정 횟수 이후에 사건이 발생 할 확률을 보고 싶을 때?(전체 시간 단위로)

## Gamma Distribution