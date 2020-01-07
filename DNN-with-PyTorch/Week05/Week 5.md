# Week 5

## 8.1.1 Deep neural Networks

***If a network has more than one hidden layer, it is called a deep neural network.***

## 8.2 Dropout

This method improves the performance of deep neural networks.

- Dropout is a popular regularization technique use exclusive for neural networks.
- For the evaluation we do not multiply the activation function with the Bernoulli random variable r. 

> Dropout : Activation 의 뉴런을 확률적으로 생략 함

```python
# Set the model to train mode
model.train()
# Set the model to evaluation mode
model.eval()
```

## 8.3 Neural Network initialization Weights

뉴런넷의 weights가 적절히 초기화 되지 않으면 정상적으로 동작하지 않음.

- 만약 모든 weight를 1 bias를 0으로 초기화 한 후 트레이닝 시킬 경우
  - 각각의 레이어가 같은 기울기를 가지고 있기 때문에 업데이트가 정상적으로 동작하지 않음

## 8.4 Gradient Descent with Momentum

- It helps to escape local optima

