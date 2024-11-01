# Credit-Card-Transactions-Fraud-Detection
### Project Overview

This project addresses the challenge of credit card fraud detection using machine learning techniques to identify suspicious transactions in real-time. With a highly imbalanced dataset, this project applies data balancing strategies, feature engineering, and hyperparameter tuning to enhance model precision.

### Data

This project uses **[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)**. This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.

- index - Unique Identifier for each row
- trans_date_trans_time - Transaction DateTime
- cc_num - Credit Card Number of Customer
- merchant - Merchant Name
- category - Category of Merchant
- amt - Amount of Transaction
- first - First Name of Credit Card Holder
- last - Last Name of Credit Card Holder
- gender - Gender of Credit Card Holder
- street - Street Address of Credit Card Holder
- city - City of Credit Card Holder
- state - State of Credit Card Holder
- zip - Zip of Credit Card Holder
- lat - Latitude Location of Credit Card Holder
- long - Longitude Location of Credit Card Holder
- city_pop - Credit Card Holder's City Population
- job - Job of Credit Card Holder
- dob - Date of Birth of Credit Card Holder
- trans_num - Transaction Number
- unix_time - UNIX Time of transaction
- merch_lat - Latitude Location of Merchant
- merch_long - Longitude Location of Merchant
- is_fraud - Fraud Flag <--- Target Class

### Key Features
- Data preprocessing: Handling missing values, scaling, and feature selection.
- Model development: Implementation of various algorithms (Logistic Regression, Random Forest, XGBoost) for classification.
- Model evaluation: Use of accuracy, precision, recall, F1-score, and ROC-AUC for performance assessment.
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV to optimize models.

### Project Workflow:
1. Data exploration and visualization
2. Preprocessing: Handling imbalanced data, feature generation and selection
3. Classification model training and testing
4. Anomaly detection  model training and testing
5. Evaluation and comparison of models
6. Conclusion 

### Data exploration and visualization

During Exploratory Data Analysis several important discoveries were done:<br>
- Dataset has strongly imbalanced data, having only 0.6% positive observations.
- All transactions made by cardholders from some cities and some zip codes were fraudulent. Not all of these zip codes belong to "fraudulent" cities.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/1.png" width="400" height="300"/>
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/2.png" width="400" height="300"/>
</p> 

- Some states have clearly more fraudulent transactions than others. State DE has only fraudulent transactions.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/3.png"/>
</p>

- In non-fraudulent transactions, there are 2 peaks at the age of 37-38 and 49-50, while in fraudulent transactions, the age distribution is a little smoother and the second peak includes a wider age group from 50-65. This does suggest that older people are potentially more prone to fraud.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/4.png"/>
</p>

- Representatives of certain jobs have only fraudulent transactions made.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/5.png"/>
</p>

- Some spending categories more frequently get fraudulent transactions, than others.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/6.png"/>
</p>

- Transactions with certain amount are more likely to be fraudulent, as we can see a clear pattern.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/7.png"/>
</p>

- Scammers activity is irregular between different days and weekdays.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/8.png"/>
</p>
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/9.png"/>
</p>

- More frauds are performed between 23:00 - 4:00, when most people sleep.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/10.png"/>
</p>