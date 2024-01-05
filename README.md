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
\large
G = \frac{(\frac{\sum_{i=1}^{n_{e}}y_{i}R(\hat{y_{i}})}{\sum_{i=1}^{n_{e}}y_{i}}) - (\sum_{i=1}^{n_{e}}\frac{n_{e}-i+1}{n_{e}})}{(\frac{\sum_{i=1}^{n_{e}}y_{i}R(y_{i})}{\sum_{i=1}^{n_{e}}y_{i}}) - (\sum_{i=1}^{n_{e}}\frac{n_{e}-i+1}{n_{e}})}
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

## Dataset
Variable Descriptions...

- **id**: *Policy key*
- **veh_value**: *Market value of the vehicle in $10,000's
exposure: The basic unit of risk underlying an insurance premium*
- **veh_body**: *Type of vehicles*
- **veh_age**: *Age of vehicles (1=youngest, 4=oldest)
gender: Gender of driver*
- **area**: *Driving area of residence*
- **agecat**: *Driver’s age category from young (1) to old (6)Driver’s age category from young (1) to old (6)*
- **engine_type**: *Engine type of vehicles*
- **max_power**: *Max horsepower of vehicles*
- **driving_history_score**: *Driving score based on past driving history (higher the better)*
- **veh_color**: *Color of vehicles
marital_status: Marital Status of driver (M = married, S = single)*
- **e_bill**: *Indicator for paperless billing (0 = no, 1 = yes)*
- **time_of_week_driven**: *Most frequent driving date of the week (weekdays vs weekend)*
- **time_driven**: *Most frequent driving time of the day*
- **trm_len**: *term length (6-month vs 12-month policies)*
- **credit_score**: *Credit score*
- **high_education_ind**: *indicator for higher education*
- **clm**: *Indicator of claim (0=no, 1=yes)*
- **numclaims**: *The number of claims*
- **claimcst0**: *Claim amount*

# Submission
CSV file in the following format:
```
id,Predict
1,0
2,200.3
3,1533.9
4,1860.2
5,80.43
6,2008.4312
```

# Neural Network Attempt
![NN Image](/src/NN/NN.png)