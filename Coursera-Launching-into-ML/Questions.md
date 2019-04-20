# Question and discussion

## 1. Why absolute cannot be used as loss function

-> For gradient descent, derivation should be able to do but, in some point, cannot derivate

Loss function  

- x-axis : weight
- y-axis : Loss function  

Gradient descent is done on the loss function plot  

## 2. Why RMSE cannot be used

->  Exactly it is not good for classification and Cross-entropy has a higher penalty when result tries to be far from target.

## 3. Regression and classification both use gradient descent

-> yes. Gradient descent is one of a good ways

## 4. gradient descent process

: input -> gradient descent ->  
If arrive minimum(cost change is not meaningful) -> return  
else  
Change weight ***(new weight = weight + (learning-rate * gradient))*** and repeat gradient descent  

## 5. why use batch

-> It lets make data parallelize and reduce training time

## 6. 10 sample of 100 is better than just 10 samples? When select samples again

-> After calculating new weight change mini batch to other mini batch and repeat gradient descent  

## 7. How mini-batch make sample parallelize

-> If mini batch size is 10 then  
(serial) -> calculate after finishing front one  
[H(10), H(9), ... ,H(1)]  ->  

(parallelize) -> calculate each one in same time  
H(1)  
H(2)  
H(3)  
.  
.  
H(10)  
Can calculate like this(parallelize)  

## 8 . Tensorflow playground -> classification? regression

- Classification : Decide class using decision boundary
- Regression : Guess data using training data

## 9. Difference many neuron vs many layer

- Many layers  
Usually many layer can make neuron advanced expression  
For example, if 1 layer only makes linear boundary dimensions, layer structure can make complex or advanced expression like a circle or spiral.  

- Many neurons
If neurons in layer are increased, expression can be variety. For example, Let's denote that specific layer only can express linear straight line, then if new neuron is added than new straight line which has a different gradient will be added in boundary decision.  

## 10. when do i have to increase neuron or layer

-> Decide a number of neurons or layers based on a paper which are suggesting a result what i want to do.  
***Good artists copy, great artists steal*** -> Impressive sentence  

## 11. What is a difference Training set, validation set and validation set

- Training set is used in training like name
- Validation set is used in checking hyper-parameter  
-> It means Validation set is used while training not after finishing. If Result of validation set is not good, then change hyper-parameter.
- Test set is used in testing. If result of test set in not good enough, then Changing model is recommended.

## 12. Bootstrapping? cross-validation

: These are used when training data is not enough,

- Bootstrapping : Regulate a proportion of data
- Cross-validation : calculate average of tests. But this test is carried out by changing test data and training data  
-> For example, If 1,2 of data was test, 3~10 of data was training data, repeat agin by setting 3,4 as test data and 1,2,5 ~ 10 as training data  

---