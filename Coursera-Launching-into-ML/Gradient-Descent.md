# Gradient Descent

```python
# pseudo code of Gradient descent
while loss is > Epsilon:
    derivative = computeDerivative()
    for i in range(self.params):
            self.params[i] = //
        self.params[i] //
            - learning_rate //
            +  stepSize * direction[i]
    if readyToUpdateLoss():
    loss = computeLoss()
```

How do i take a loss function adn turn it into a search strategy?  

- Gradient Descent refers to the process of walking down the surface formed by using our loss function on all the points int parameters space.  

## Problem

- Which direction should i head?
- How far away should i step?  

How the algorithm takes fixed size steps in the direction od the minimum

- If step size is too small, training might take forever.
- If step size is too big, might either bounce from wall to wall of loss surface or bounce out of the valley entirely and into an entirely new part of the loss surface -> cannot guarantee to converge.  

---

## Troubleshooting a Loss Curve

Learning rate is likely to have a problem specific best value. Because it is said before learning begins, learning the rate is a hyperparameter  

And to determine the best value for hyper-parameters, there is a better method available and it is called hyperparameter tuning.

---

## ML Model pitfalls

In ML, sometimes programmers are not used to working in deterministic settings.  
For many models, if you retrain the model a second t ime, even when using the same hyper-parameter settings, the resulting parameter settings might be very different. This at first seems disconcerting, aren't we looking for the best set of parameters? Does this mean that gradient descent isn't working? Or that i'd implemented incorrectly?  Does this mean that gradient descent isn;t working? Or that I'd implemented incorrectly? Not necessarily
-> ***Because of local minimum***(Because of non-convex surface)

### Three prime steps our algorithm must go through

Loop  

1. Calculate Derivative
2. Take a Step
3. Check Loss
4. If done, escape loop or not loop again

### What can we change to improve training time

Typically, the number of effective parameters in a model is fixed, although we will return to how this might be varied in a future module on regularization. Additionally, although it might sound appealing to ***reduce the number of data points used to check the loss, this is generally not recommended.***  

***We have two main knobs***

The number of data points we calculate the derivative on and the frequency with which we check the loss.

Knobs for turing to speed up model training

1. The number of data points that we calculate the derivate on.(Decrease complexity) -> reduce the number of data points that we feed into our loss function or each iteration of our algorithm.

Why it works well?  
The reason that might still work is that it's possible to extract samples from our training data, then on average balance each other out.(Kind of sampling)  
Our sampling strategy selects from our training set with uniform probability. So every instance in the training set has an equal chance of being seen by the model.
***-> It is called as mini batching***  

- The samples themselves are refereed to as Batches.
- Mini-batch gradient descent has the added benefit in addition to costing last time of using less memory and of being easy to parallelize.
- Batch gradient descent compute the gradient on the entire dataset. But in tensorflow mini-batch is called as batch.  
- Batch size is another hyper-parameter like the learning rate.

2. Frequency with which we check the loss.

Recall that although it would be great to simply check the loss on a subset of the data, this isn't a good idea. Some popular strategies for the ready to update loss function are time-based and step-bases. ***For example, once every 1,000 steps or once every 30 minutes.***
