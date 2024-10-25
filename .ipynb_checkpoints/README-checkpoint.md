# Credit-Card-Transactions-Fraud-Detection
## Project Overview
  
As online transactions increase, so does the risk of fraud, making effective detection systems crucial for financial institutions.

In this project, we leverage various algorithms and data processing techniques to analyze transaction data and flag potential fraud. You will find detailed documentation on data preprocessing, model selection, and evaluation metrics to help you understand how these systems work in real-world applications.

## Data

This project uses **[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)**. This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

-index - Unique Identifier for each row
-trans_date_trans_time - Transaction DateTime
-cc_num - Credit Card Number of Customer
-merchant - Merchant Name
-category - Category of Merchant
-amt - Amount of Transaction
-first - First Name of Credit Card Holder
-last - Last Name of Credit Card Holder
-gender - Gender of Credit Card Holder
-street - Street Address of Credit Card Holder
-city - City of Credit Card Holder
-state - State of Credit Card Holder
-zip - Zip of Credit Card Holder
-lat - Latitude Location of Credit Card Holder
-long - Longitude Location of Credit Card Holder
-city_pop - Credit Card Holder's City Population
-job - Job of Credit Card Holder
-dob - Date of Birth of Credit Card Holder
-trans_num - Transaction Number
-unix_time - UNIX Time of transaction
-merch_lat - Latitude Location of Merchant
-merch_long - Longitude Location of Merchant
-is_fraud - Fraud Flag <--- Target Class

## Methodology
- Data preprocessing: Handling missing values, scaling, and feature selection.
- Model development: Implementation of various algorithms (Logistic Regression, Random Forest, XGBoost) for classification.
- Model evaluation: Use of accuracy, precision, recall, F1-score, and ROC-AUC for performance assessment.
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV to optimize models.

## Objectives:
-Analyze the data of 1.3 million transactions
-Find out hidden associations between fraud and transaction features
-Perform feature generation and bala detect fraudent transactions

## Data exploration and visualization

**Objective:**
The objective of this analysis is to 
utilize non-parametric and semi-parametric methods of survival analysis to answer the following questions.
- How the likelihood of the customer churn changes over time?
- How we can model the relationship between customer churn, time, and other customer characteristics?
- What are the significant factors that drive customer churn?
- What is the survival and Hazard curve of a specific customer?
- What is the expected lifetime value of a customer?

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/1.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/5.png" width="400" height="300">
</p>

**Cardholders**
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/4.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/9.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/7.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/15.png" width="400" height="300">
</p>

**Transaction time**
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/2.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/12.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/11.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/3.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/10.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/13.png" width="400" height="300">
</p>

**Transactions**
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/6.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/8.png" width="400" height="300">
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/14.png" width="400" height="300">
</p>

## Process
1. 
2. Preprocessing: Handling imbalanced data using SMOTE
3. Model training and testing
4. Evaluation and comparison of models
5. Conclusion 