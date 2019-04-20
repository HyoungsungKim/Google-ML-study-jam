#Introducing to Loss Functions

***Learn how to***

- Use Loss Functions

---

How we can think of optimizing our parameters, as searching through parameter space  

Loss functions : Loss functions are able to take the quality of predictions for a group of data points from our training set and compose them into a single number, with which to estimate the quality of the models current parameters.   One measure of the quality of the prediction at a single point, is simply this sign difference between the prediction and the actual value. This difference is called the ***error***

- How might we put a bunch of error values together? -> compose them as a sum  

However, if we were to use the sum function to compose our terms, the resulting model would treat error terms of opposite sign as canceling each other out.

> For example If there are -1 and 1 error in bunch  
> It will be offset.  

- The sum of the absolute values of the error seems like a reasonable alternative, ***but there are problems with this method of composing data as well, which we will tackle shortly***

Instead, what is often used as what is called the ***Mean Squared Error***  

- Mean Squared Error(MSE) : MSE is computed by taking the set of error terms from our dataset. ***Taking their squares*** to get rid of the negatives, and ***computing the average of the squares***.

The MSE is perfectly valid loss function, but it has one problem.  
Although errors might be in *pounds*, or *kilometers*, or *dollars*, the square error will be in *pounds squared*, *kilometers squared*, or *dollars squared*.  
-> That can make the MSE kind of hard to interpret  

$$sqrt(sum_{i = 1}^n (\hat Y_i - Y_i)^2)$$  

Therefore, we often take the square root of the MSE instead to get units that we can understand RMSE.  Y with ^(hat) means prediction.
-> So What we want to do is to minimize RMSE.

- Although RMSE works fine for linear regression problems, it doesn't work as a loss function for classification.  
- Classification problems are ones in which the label is a categorical variable. -> for categorical variables binary integers are used.  
- ***RMSE cannot cover when label is zeors*** -> We need other loss function
  
One of the most commonly used loss function for classification is called ***Cross Entropy*** or ***log loss***

- Cross entropy penalizes bad predictions very strongly  

