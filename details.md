# 2023 Travelers Analytics Case Competition
[Kaggle Link](https://www.kaggle.com/competitions/2023-travelers-university-competition/overview)

## Task

1. Create model for predicting claim cost for each policy
    - Predict claim cost for each policy
    - Submit CSV file of prediction for validation data
    - **Goal**: Outperform **LightGBM** model in terms of the **Gini Index**


2. Create presentation about results & explanation of model
    - Presentation for **non-statistician** business partner!
    - Areas to address
        - What method did you choose in the end and why?
        - How did you do the variable selection?
        - What variables help explain pure premium?
        - What other variables not in the data set do you think might be useful?
        - How did you test the assumptions of this method?
        - How did you evaluate your model (e.g. fit statistics, over-fitting, etc.)?
        - Any concerns about the resulting model?
        - What questions to you have about the data?

# Further Context (for me)
## LightGBM Model
Is a decision tree that grows leaf-wise by choosing a leaf it believes will yield the largest decrease in loss.
[![LightDMG Diagram](/images/LightGBM%20Diagram.jpeg)](https://ilkerkatkat.blogspot.com/2020/05/python-light-gbm.html)
[![LightDMG vs XGBoost](/images/LightGBM%20vs%20XGBoost.jpeg)](https://ilkerkatkat.blogspot.com/2020/05/python-light-gbm.html)
Decision trees split the data based on the features in order to predict some numerical value which can be used in case of both regression and classification. LightGBM uses an ensemble of decision trees because a single tree is prone to overfitting. The split depends upon the entropy and information-gain which basically defines the degree of chaos in the dataset

## Gini Index
- Is a measure of statistical dispersion. Typically used in decision-trees. Measure of a model's ability to rank predictions correctly, where higher value indicated better model performance.
- Range [0, 1]
    - 0 = all data points in a node belong to the same class ("pure")
    - 1 = all data points in a node is evenly distributed among various classes ("impure")

$$
\LARGE
G = \frac{\frac{\sum_{i=1}^{n_{e}}y_{i}R(\hat{y_{i}})}{\sum_{i=1}^{n_{e}}y_{i}} - \sum_{i=1}^{n_{e}}\frac{n_{e}-i+1}{n_{e}}}{\frac{\sum_{i=1}^{n_{e}}y_{i}R(y_{i})}{\sum_{i=1}^{n_{e}}y_{i}} - \sum_{i=1}^{n_{e}}\frac{n_{e}-i+1}{n_{e}}}
$$

- $y_{i}$ := true claim cost
- $\hat{y_{i}}$ := predicted claim cost
- $n_{e}$ := number of examples
- $R(s_{i})$ := function calculates the rank of a value within the sequence $\{ {s_{1}, ... , s_{n_{e}}} \}$ where... 
    - IF $s_{i} < s_{j}$ THEN $R(s_{i}) < R(s_{j})$
    - IF $s_{i} = s_{j}$ & $i > j$ THEN $R(s_{i}) > R(s_{j})$
        - Basically assume given list of claims cost sorted from lowers to highest. We need to rank them predicted cost in the order from lowest to highest.
        - We also need to ensure that if $s_{i} = s_{i+1}$ then $Rank(s_{i}) < Rank(s_{i+1})$ to break any ties.
    
- $\Large \frac{\sum_{i=1}^{n_{e}}y_{i}R(\hat{y_{i}})}{\sum_{i=1}^{n_{e}}y_{i}}$ = represents the fraction of models predicted claim cost vs true claim cost, by summing up products fo actual costs and their predicted rankings.
- $\Large \sum_{i=1}^{n_{e}}\frac{n_{e}-i+1}{n_{e}}$ = calculates the sum of a series of decreasing values, representing the ideal case of ranking.
- Benchmark score = 0.18029