# Credit-Card-Transactions-Fraud-Detection
## Project Overview
  
As online transactions increase, so does the risk of fraud, making effective detection systems crucial for financial institutions.

In this project, we leverage various algorithms and data processing techniques to analyze transaction data and flag potential fraud. You will find detailed documentation on data preprocessing, model selection, and evaluation metrics to help you understand how these systems work in real-world applications.

## Data

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

## Methodology
- Data preprocessing: Handling missing values, scaling, and feature selection.
- Model development: Implementation of various algorithms (Logistic Regression, Random Forest, XGBoost) for classification.
- Model evaluation: Use of accuracy, precision, recall, F1-score, and ROC-AUC for performance assessment.
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV to optimize models.

## Objectives:
- Analyze the data of 1.3 million transactions
- Find out hidden associations between fraud and transaction features
- Perform feature generation and balance data for training classification models
- Build a Machine Learning model to predict which previously purchased product will be in user’s next order

## Data exploration and visualization

Classification model performance is strongly dependant on balance of target classes in data it is trained on. As the graph below shows, provided dataset has strongly imbalanced data. This problem will be addressed later, during preprocessing data for model training.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/1.png" width="450" height="335">
</p>
Data has no obvious relationships between it's features and target. Strongest correlation with target has feature amt.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/5.png" width="800" height="600">
</p>

**Cardholders**
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/4.png" width="400" height="300">
</p>
- Most cardholders are of age between 38 and 62 years.
- Distribution inside most numerous group is uneven with drop at 43 years.
- There are no cardholders younger than 20 years. 
- Group 20-38 has same amount as group 62-100.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/9.png" width="400" height="300">
</p>
The age distribution is visibly different between 2 transaction types. In non-fraudulent transactions, there are 2 peaks at the age of 37-38 and 49-50, while in fraudulent transactions, the age distribution is a little smoother and the second peak includes a wider age group from 50-65. This does suggest that older people are potentially more prone to fraud.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/7.png" width="400" height="300">
</p>
In this case, we do not see a clear difference between both genders. Data seem to suggest that females and males are almost equally susceptible (50%) to transaction fraud. Gender is not very indicative of a fraudulent transaction.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/15.png" width="600" height="500">
</p>
Fraudulent transactions are performed is different places, without clusters.

**Transaction time**
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/2.png" width="400" height="300">
</p>
There are seasonal increases of total amount of transactions.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/12.png" width="400" height="300">
</p>
Distribution of fraudulent transactions doesn't correlate to distribution of non-fraudulent transactions. While normal payments peak in December and then March-July, fraudulent transactions are more concentrated in Jan-May. 
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/11.png" width="400" height="300">
</p>
Relative amount of non-fraudulent transactions increases on Monday and Sunday, while relative amount of fraudulent transactions is distributed more evenly.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/3.png" width="400" height="300">
</p>
Most transactions are performed after 13:00
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/10.png" width="600" height="450">
</p>
Non-fraudulent transactions are more or less evenly distributed, without noticeable peaks. Fraudulent transactions have peaks at 22-04 hours, when most people are asleep.

**Transactions**
On this graph only transactions less than 1000USD are displayed, due to big rane of amt and insignificant amount of such transactions
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/6.png" width="400" height="300">
</p>
While normal transactions tend to be around $200 or less, we see fraudulent transactions peak around $300 and then at the $800-$1000 range.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/8.png" width="600" height="300">
</p>
Fraud tends to happen more often in 'Shopping_net', 'Grocery_pos', and 'misc_net' while 'home' and 'kids_pets' among others tend to see more normal transactions than fraudulent ones.
<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/14.png" width="700" height="450">
</p>
Distribution of fraudulent and non-fraudulent transactions between different spending categories doesn't correlate, showing there are some preffered categories by scammers.

## Process
1. 
2. Preprocessing: Handling imbalanced data using SMOTE
3. Model training and testing
4. Evaluation and comparison of models
5. Conclusion 