# Generalization and Sampling

When is the most accurate ML model not the right one to pick?  
-> When a model has a loss metric of zero for your training dataset does not mean that it will perform well on new data out there in there real world(Overfitting)  

***The best ML model is not necessarily the one that performs best on just your training dataset, but it is the one that performs best on unseen data.***

Learn how to  

- Assess if your model is overfitting  
- Gauge when to stop model training  
- Create repeatable training, evaluation and test datasets
- Establish performance benchmarks

---

## Generalization and ML Models

Complex model could have more free parameters.  
In ML, we often have lots of data and no such intuition. one of the best ways to assess the quality of a model is to see how it performs well against a new dataset that it hasn't seen before. Then we can determine whether or not that model generalizes well across new data points.  

***How can i make sure that my model is not overfitting?***
***How do i know when to stop training?***  
-> Answer is splitting data.(Training & Validation)  

- Underfitting : There is quite a few points outside the shape of that trend line.
- Overfitting : Right extreme.

Overfitting or memorized in that training dataset can be often far worse than having a model that only adequately fits your data.

---

## When to stop Model Training

Should i use linear regression or a neural network?  You can also use your validation data-set to help fine tune those hyper-parameters of a single model, which if you recall, those hyper-parameters are set before training.  

This tuning process is accomplished through successive training runs, and then comparing those training runs against that independent validation dataset to check for overfitting.  

Training the model is where we start to calculate random weights, calculate that derivative, look at the direction down the gradient descent loss curve, minimize your loss metric and then repeat.  

When i want to assess the performance of a model against data that has not yet seen in training, which is where we use the validation dataset.  

And if there is not a significant divergence between the loss metrics from the training run, and the loss metrics for the validation dataset run , then we could potentially go back and optimize our hyper-parameters a little bit more.  

```txt
Process

Start with a NN model with one ser of parameters
Train model using training dataset
Is it overfitting?(evaluate using validation dataset)
if yes -> Stop and Use model with hyper-parameters of last non-overfit model for prediction
if no -> Increase model complexity and repeat Train using training dataset
```

Splitting data into three parts

- Training
- Validation
- Brand new completely isolated silo called ***testing***

What happens if you fail on your testing dataset, even though you passed validation? Means you cannot retest the same ML model, and you have got to either retrain a brand new machine Learning model, or go back to the drawing board and collect more data samples to provide new data for ML model.  

- Dataset : experimental + Test
- Experimental : Training + validation -> bootstrapping or cross-validation

If have a lot of data, use the approach of having a completely independent held-out test dataset, that is like go or no-go decision  
If Don't have that much data, use the cross-validation approach.  

---

## Sampling 

Set 80 percent of the data are in the training data set and the others are in test and validation.