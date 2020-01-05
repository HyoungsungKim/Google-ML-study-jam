# Week4 

## 4.1 Multiple Linear regression Prediction

### Multiple Linear Regression(MLR)

In Multiple linear regression we have multiple predictor variables, in this example we have 4 predictor variables,
$$
\hat{y} = b_0 + w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4
$$

- We can express the operation as a linear transformation

### Linear Regression

1. x is a 1D tensor and is sometimes referred to as a feature
2. w is a D by 1 tensor or vector of parameters  y-hat is the dependent variable. 

x : 1 * D, w= D * 1
$$
x = [x_1, x_2 ... , x_D] \\
w = [[w_1], [w_2], [w_3], ... , [w_D]] \\
\hat{y} = x*w + b
$$

- ***x\*w is scholar***

### 2D Tensors/Matrices

$$
x_1 = [4.9, 3, 1.4, 0.2]\\
x_2 = [4.1, 1, 1.4, 0.2]\\
x_3 = [1.1, 2.1, 3, -1] \\
x_4 = [4.3, 1.9, 1, 7.9] \\
X = [[x_1], [x_2], [x_3], [x_4]] \\
\hat{y} = Xw + b
$$

```python
from torch.nn import Linear
torch.manual_seed(1)
#yhat = x*w + b
model = Linear(in_features = 2, out_features = 1) 
list(model.parameters())
'''
output:
tensor([[ 0.3643, -0.3121]], requires_grad=True), Parameter containing:
tensor([-0.1371], requires_grad=True)]
'''
model.state_dict()
'''
output:
OrderedDict([('weight', tensor([[ 0.3643, -0.3121]])), ('bias', tensor([-0.1371]))])
'''
X = torch.tensor([1.0, 3.0])
yhat = model(x)
```

Example

```python
import torch.nn as nn
class LR(nn.Module):
   def __init__(self, input_size, output_size):
    super(LR, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    
   def forward(self, x):
    out = self.linear(x)
    return out

class LR2(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR2, self).__init__()
        self.network = nn.Sequential()
        network.add_module('layer1', nn.Linear(input_size, output_size))
        
    def forward(self, x):
        out = self.netword(x)
        return out       
```

## 5.0 Linear Classifiers

### Sigmoid

- If the value of Z is a very large negative number the expression is approximately 0.
- And for a very large positive value of Z the expression is approximately 1.
- And for everything in the middle the value is between 0 and 1. 
  - If the points are very close to the center of the line, the value for the sigmoid function is very close to 0.5, this means that we are not certain if the class is correct.
  -  if the points are far away, the value for the sigmoid function is either 0 or 1, respectively this means we are certain about the class. 

## 5.1 Logistic Regression: Prediction

What's the difference between a linear regression custom module  and a logistic regression custom module?

- We apply the logistic function

## 5.2 Bernoulli Distribution

Bernoulli Distribution
$$
p(y|\theta) = \theta^y(1-\theta)^{1-y}
$$
Log likelihood (Use Bernoulli Distribution)
$$
\begin{align}
\mathcal{l}(\theta) & = ln(p(Y|\theta)) \\
& = \sum^N_{n = 1} y_nln(\theta) + (1 - y_n)ln(1 - \theta)
\end{align}
$$

## 5.3 Logistic Regression Cross Entropy Loss

- MSE : If initialization is bad, then cannot be converged
- Cross Entropy Loss : There is only one minimum
  - If multiply -1 to maximum likelihood, then we can find minimum