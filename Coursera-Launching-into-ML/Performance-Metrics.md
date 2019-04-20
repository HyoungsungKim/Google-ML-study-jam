# Performance Metrics

Learn how to

- Choose Effective Performance Metrics

---

The models that we created(in TensorFlow Playground), were able to learn complex non-linear relationships using a learned hierarchy of features.  
However, we discovered at the end of the section that our current approach suffers form problems The consequence of which include things like long training times, suboptimal minima, and inappropriate minima.

## How performance metrics help us get better result when there are inappropriate minimum  

- An inappropriate minimum : points in parameter space that reflect strategies, either that won't generalize well, that don't reflect the true relationship being modeled or both.

When datasets are skewed and contain far more of one class than another, then suddenly strategies like this can become much more seductive.  

We need to reframe the problem. Instead of searching for the perfect loss function during training, we're instead going to use a new sort of metric after training is complete. And this new sort of metric will allow us to reject models that have settled into inappropriate minima. ***Such metrics are called performance metrics.***

 Loss Function | Performance Metrics
---------------|----------------------
- During training | - After training
- harder to understand | - Easier to understand
- Indirectly connected to business goals | - Directly connected to business goals

Example of Loss function : Cross entropy, RMSE (loss metrics)
Example of Performance Metrics : Precision, Accuracy, Recall(True positive / (False positive + True Positive)) (performance metrics)
Performance metrics have two benefits over loss functions.

- They are easier to understand.
- performance metrics are directly connected to business goals.

---
