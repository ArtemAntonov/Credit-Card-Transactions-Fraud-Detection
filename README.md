[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)

# Credit-Card-Transactions-Fraud-Detection
### Project Overview

This project tackles the challenge of detecting fraudulent credit card transactions using machine learning. It leverages feature engineering, dimensionality reduction(MCA and polynomial features), model selection(KNN, Decision Tree, Random Forest, MLP) and advanced hyperparameter optimization techniques like HalvingGridSearchCV to build a robust fraud detection system.

### Data

This project uses **[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)**. This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020. It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants.
<details>
<summary>columns description</summary>
    
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
</details>

### Key Features
- **Data preprocessing:** Handling outliers, feature generation, feature selection, dimensionality reduction and scaling.
- **Model development:** Implementation of various algorithms for classification.
- **Model evaluation:** Use of accuracy, precision, recall and F1-score for performance assessment.
- **Hyperparameter tuning:**  HalvingSearchCV and GridSearchCV to optimize models.

### Project Flow:
1. [Exploratory Data Analysis](https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/#1-exploratory-data-analysis)
2. [Preprocessing: feature generation and selection, handling imbalanced data](https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/#2-preprocessing)
3. [Model training and testing](https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/#3-model-training-and-testing)
4. [Conclusion](https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/#4-Conclusion)

### 1. Exploratory Data Analysis

During Exploratory Data Analysis several important discoveries were done:<br>
- Dataset has strongly imbalanced data, having less than 0.6% positive observations.
- Some states have clearly more fraudulent transactions than others. State DE has only fraudulent transactions.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/3.png"/>
</p>

- In non-fraudulent transactions, there are 2 peaks at the age of 37-38 and 49-50, while in fraudulent transactions, the age distribution is a little smoother and the second peak includes a wider age group from 50-65. This does suggest that older people are potentially more prone to fraud.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/4.png"/>
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
- More frauds are performed between 23:00 - 4:00, when most people sleep.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/10.png"/>
</p>

- Transactions of certain amounts are fraudulent among some age groups.
- In some states fraud transactions are made among holders of cards by specific issuers' industry codes.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/13.png" width="460"/>
</p>

- Depending on population, there's higher probability of fraud based on card issuer's industry code.

### 2. Preprocessing

Based on exploratory data analysis, initial dataset has been changed:
- New features created: day, weekday, hour, industry_code and age.
- Not useful features were deleted: time, first, last, street, trans_num, unix_time, cc_num, dob, lat, long, merch_lat, merch_long, zip, city, job, merchant.
- Outliers were corrected by imputation with arbitrary value.

Resulting data set contained numeric and categorical features, which were treated separately. Categorical features were processed by MCA with 68 components resulting in approx. 100% variability saved. Polynomial features were generated from numerical attributes and evaluated using Lasso regularization to identify the most informative interaction. Best performing set had degree 2. Top 60 most important features were selected using Lasso regression.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/20.png" width="400"/>
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/21.png" width="400"/>
</p> 

To improve target class balance in the dataset, several sampling algorithms were tested by different models. Unfortunately no performance gain was achieved by any resampling method. Also, overfitting on test data was introduced by all resamplers, except EditedNearestNeighbours.<br/>
Best scores were achieved on EditedNearestNeighbours processed data and data without resampling. Best performing models: Decision Tree, Nearest Neighbour, MLPC and Random Forest.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/22.png"/>
</p>

Resampling increased models' training time and introduced overfitting without noticeable improvement of F1 score, so no resampling method was used further.<br/>
Data preparation pipeline including all steps for data preprocessing was created as a result of this section.

### 3. Model training and testing

As the dataset has severe class imbalance, any model that always outputs "Not fraud" will score 99.4% accuracy. That's why F1 score was used during model training.<br/>
The dataset was split into training and testing sets, with 75% of the data used for training.<br/>
For initial detection of promising hyperparameters space HalvingGridSearchCV was used. Then GridSearchCV was used to select best hyperparameters from the area. <br/>
Repeated Stratified K-Fold cross validation was used to validate performance and avoid overfitting.<br/>
To track the progress, training results were stored in a dataframe and displayed as plots for each model highlighting the best F1 score iteration on the test set. Grid search results were stored as well in order to allow retrieving trained models and analyze results of each grid search. Some files were too big for Github, so they were archived in "saves" folder.<br/><br/>

KNeighbors classifier showed the biggest increase in F1 score after hyperparameters tuning - 0.0949, but it didn't even reach the scores of other models with default hyperparameters. 

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/30.png" width="460"/>
</p>

Decision Tree model achieved second result in F1 score increase during hyperparameters tuning(0.0393). Models were trained very fast, allowing performing of greater amount of experiments than for KNN. Unfortunately, the score of best iteration didn't reach base scores of Random Forest and MLPC.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/31.png" width="460"/>
</p>

Random Forest showed lesser than previous models increase in F1 score during hyperparameters tuning  - 0.0203, but combined with high basic score of 0.8369 it showed the best result among trained models - 0.8572. Training models took significant time, reducing the amount of possible experiments(time limitation). The model F1 score reaction on hyperparameters' changes wasn't always linear, leaving possibility that better hyperparameters can be found.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/32.png" width="460"/>
</p>

Multi-layer Perceptron classifier showed same increase of F1 score after during tuning(0.0215), but it had lower initial score, so it couldn't outperform Random Forest, scoring max 0.8489 F1 score on test data. Best scoring model was neither the widest or the deepest among tested.<br/>

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/33.png" width="460"/>
</p>

After selecting the model with best performance(Random Forest of 11 iteration), a pipeline was developed, containing all data preprocessing steps and the selected classifier.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/34.png" width="300"/>
</p>

Developed pipeline was tested on 100% previously unuseen data, achieving 0.8 F1 score.

### 4. Conclusion 

In this project a machine learning pipeline that combined preprocessing, feature engineering, dimensionality reduction and classification was created. This pipeline scored 0.86 F1 score on test data and 0.8 on unseen data.<br/>
To handle class imbalance of data, several resampling(oversampling and undersampling) methods were used, but none of them showed improvement in models' scores. This shows that resampling doesn't always solve the problem of unbalanced data.<br/>
Several models(KNN, Decision Tree, Random Forest, MLPC) had their hyperparameters tuned to find the best performer. Random Forest outperformed MLPC, indicating that deep learning models are not always an optimal solution.<br/>
Among MLPC models, best performer didn't have the widest or biggest amount of hidden layers, showing that increasing this values doesn't automatically lead to better results.