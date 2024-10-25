# Credit-Card-Transactions-Fraud-Detection
## Project Overview
  
As online transactions increase, so does the risk of fraud, making effective detection systems crucial for financial institutions.

In this project, we leverage various algorithms and data processing techniques to analyze transaction data and flag potential fraud. You will find detailed documentation on data preprocessing, model selection, and evaluation metrics to help you understand how these systems work in real-world applications.

## Data

This project uses **[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)** which contains generated credit card transactions made by 1000 cardholders. It comprises over 1.8 million trasnsactions. 

## Features
- Data preprocessing: Handling missing values, scaling, and feature selection.
- Model development: Implementation of various algorithms (Logistic Regression, Random Forest, XGBoost) for classification.
- Model evaluation: Use of accuracy, precision, recall, F1-score, and ROC-AUC for performance assessment.
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV to optimize models.

## Process
1. Data exploration and visualization
2. Preprocessing: Handling imbalanced data using SMOTE
3. Model training and testing
4. Evaluation and comparison of models
5. Conclusion 

## Results




## Customer Survival Analysis

**Survival Analysis:** 
Survival analysis is generally defined as a set of methods for analyzing data where the outcome variable is the time until the occurrence of an event of interest. The event can be death, occurrence of a disease, marriage, divorce, etc. The time to event or survival time can be measured in days, weeks, years, etc.

For example, if the event of interest is heart attack, then the survival time can be the time in years until a person develops a heart attack.

**Objective:**
The objective of this analysis is to utilize non-parametric and semi-parametric methods of survival analysis to answer the following questions.
- How the likelihood of the customer churn changes over time?
- How we can model the relationship between customer churn, time, and other customer characteristics?
- What are the significant factors that drive customer churn?
- What is the survival and Hazard curve of a specific customer?
- What is the expected lifetime value of a customer?

**Kaplan-Meier Survival Curve:**

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/download.png" width="400" height="300">
</p>

From above graph, we can say that
- AS expected, for telcom, churn is relatively low. The company was able to retain more than 60% of its customers even after 72 months.
- There is a constant decrease in survival probability probability between 3-60 months.
- After 60 months or 5 years, survival probability decreases with a higher rate. 