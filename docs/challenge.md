Model selected: Logistic Regression with Feature Importante and with Balance

Explanation:
The goal was to develop a model that predicts the probability of a flight experiencing a delay at SCL airport. The optimal model would therefore be the one that most accurately predicts delays (Class 1: Delay = True). Among the four models evaluated, "Logistic Regression with Feature Importance and Balance" and "XGBoost with Feature Importance and Balance" emerged as the top contenders.

Both models demonstrated superior performance in classifying delays, as indicated by their recall scores. Recall is crucial for minimizing false negatives (missed delays), and models without class balancing achieved a recall of just 0.01, meaning they missed 99% of the delays, which made them unsuitable.

The performance metrics for XGBoost and Logistic Regression in predicting Class 1 are comparable. Specifically, they have a precision of 0.25, meaning that 25% of the predicted delays are correctly identified. Their recall is 0.69, indicating that it successfully identifies 69% of actual delays. Both models have an accuracy of 0.55, reflecting that they correctly predict the outcome 55 times out of every 100 in the test dataset samples. Although XGBoost has a slightly higher F1-score (a difference of 0.01), Logistic Regression was selected over XGBoost due to its interpretability. Logistic Regression's coefficients provide clear insights into how each feature affects the probability of a delay, making the model's results easier to understand and explain.

Logistic Regression offers valuable interpretative advantages. The model provides clear insights into the relationship between features and the output through its coefficients, which can be directly translated into odds ratios. This helps in understanding how changes in individual features impact the likelihood of a delay.

Logistic Regression allows for hypothesis testing to determine the statistical significance of the model coefficients. This feature is essential for identifying which predictors are most influential in predicting delays. Additionally,  it is simpler and computationally less intensive compared to XGBoost.



              precision    recall  f1-score   support

           0       0.88      0.52      0.65     18294
           1       0.25      0.69      0.36      4214

    accuracy                           0.55     22508
   macro avg       0.56      0.60      0.51     22508
weighted avg       0.76      0.55      0.60     22508
