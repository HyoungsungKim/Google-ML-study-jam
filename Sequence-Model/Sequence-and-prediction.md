# Sequence and prediction

How do you split time series data into training, validation, and testing sets? 

### Train, validation and test sets

- ***We typically want to split the time series into a training period,  validation period and a test period.*** This is called fixed partitioning. If the time series has some seasonality, you generally want to ensure that each period contains a whole number of seasons. For example, one year, or two years, or three years, if the time series has a yearly seasonality
  - Fixed partitioning
    - You'll train your model on the training period,
    - you'll evaluate it on the validation period. 
  - Roll-forward partitioning
    - We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time.
    - At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period.

### Moving average and differencing

- A common and very simple forecasting method is to ***calculate a moving average.***
  - This nicely eliminates a lot of the noise and it gives us a curve roughly emulating the original series, ***but it does not anticipate trend or seasonality.***
    - Depending on the current time i.e. the period after which you want to forecast for the future, it can actually end up being worse than a naive forecast.
  - In this case, for example, I got a mean absolute error of about 7.14.
    - One method to avoid this is to remove the trend and seasonality from the time series with a technique called differencing.
    - So instead of studying the time series itself, ***we study the difference between the value at time T and the value at an earlier period.***
  - Depending on the time of your data, that period might be a year, a day, a month or whatever.
    - Let's look at a year earlier.
    - So for this data, at time T minus 365, we'll get this difference time series which has no trend and no seasonality.
    - We can then use a moving average to forecast this time series which gives us these forecasts. But these are just forecasts for the difference time series, not the original time series.
      - T에서 365를 뺀 값으로 무빙 에버리지 계산 함
      - 원본 time series에 관한 건 아님
    - ***To get the final forecasts for the original time series, we just need to add back the value at time T minus 365, and we'll get these forecasts.***
      - If we measure the mean absolute error on the validation period, we get about 5.8. So it's slightly better than naive forecasting but not tremendously better.
      - You may have noticed that our moving average removed a lot of noise but our final forecasts are still pretty noisy. Where does that noise come from?
        - ***That's coming from the past values that we added back into our forecasts.***
          - 원래 이동 평균에 비해 노이즈가 존재하는데, 이전에 difference에 값을 더하는 과정에서 노이즈도 같이 더해졌기 때문에 발생한 노이즈임.
        - So ***we can improve these forecasts by also removing the past noise using a moving average on that.***
          - 따라서 노이즈가 있는 값에 한번 더 이동 평균 적용
        - If we do that, we get much smoother forecasts.
        - In fact, this gives us a mean squared error over the validation period of just about 4.5. Now that's much better than all of the previous methods.
    -  Moving averages using centered windows can be more accurate than using trailing windows.
      - But we can't use centered windows to smooth present values since we don't know future values.
      - However, to smooth past values we can afford to use centered windows. 

