# Google ML study jam coursera lecture note

## Introduction to Optimization

Learn how to

- Quantify model performance using loss functions
- Use loss functions as the basis for an algorithm called gradient
- Optimize gradient descent to be as efficient as possible
- Use performance metrics to make business decisions

---

## Defining ML Models

Optimization always requires a standard by which to say we are improving -> loss function  
How gradient descent is like trying to find the bottom of a hill defined by the loss function?  

### Reviewing exactly what an ML model is and where parameters fit into the equation

ML models are mathematical functions with parameters and hyper-parameters.

- Parameter         : Real-valued variable that changes during training
- Hyper-parameter   : Setting that we set **before training** and which doesn't change afterwards

Linear models were one of the first sorts of ML models. They remain an important and widely used class of models today though.  

In a linear model, **small changes in the independent variables, or features** as we refer to them in machine learning, **yield the same amount of change in the dependent variable or label**.  

$$ y = x*m + b $$  
y : output  x : input  
m : weight  b : bias
m captures the amount of change we've observed in our label in response to a small change in our future.

$$ y = X*w + b$$

When we increase the dimensionality of the input, our slope term m, must become n-dimensional. We call this new term the weight. -> this process yields the n-dimensional generalization of a line, which is called a hyper-plane

- Make n-dimensional using vector
- Hyper-plane : Plot of n-dimensional linear model  
- b term is a scalar or vector -> referred to as the bias term.

Gow cna a linear model be used for classification? How do you take a continuous number and interpret it as a class? -> need encoding  
The simples way to encode class membership is with a binary. Either you are a member, or you are not.

> Binary means **0(false) or 1(true)**
> for example, this is cat or not

Of course in many cases categorical variables can take more than two values using the decision boundary.  
The decision boundary is intended not just to be descriptive of the current data. It is intended to be predictive of unseen data.  
***This property of extending to unseen examples is called generalization, and it is essential to ML models***

### Which one can be a ML problem

Which one could be a label in our model?  
***What i should have intuitions about are when the information is available relative to when we want to actually make predictions.***
If we selected label, we have to consider it is continuous  

> Classification and Regression which one?  
> It is decided by output type  
>   
> Classification    -> For predicting discrete class label  
> Regression        -> For predicting continuous quantity  
> 
> Example)  
> Predicting a price of house -> Regression (output is 0 ~ 1,000,000,000)  
> Predicting a spam mail or not -> Classification (output is yes(1) or no(0))

Often the first step to modeling data is to look at the data to ***verify that there is some signal and that it's not all noise.***

- Underfitting : Model a non-linear function with a linear model is called underfitting.
  
***How do we know which criterion is good?***

- ***Least squares regression*** : Process for determining the best weights

But It is only work up to a certain scale. For example ***Once start using really big data sets***, the computation required to analytically solve this problem becomes impractical.  

Need ***Gradient descent***  

- What loss function do? : We will need to somehow generalize from the quality of a prediction for a single data point, which is simply the error of that prediction, to a number that captures the quality of a group of predictions.

---

