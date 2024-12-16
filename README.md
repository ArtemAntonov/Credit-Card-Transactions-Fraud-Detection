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
- **Data preprocessing:** Handling outliers, feature generation, feature selection and scaling.
- **Model development:** Implementation of various algorithms for classification.
- **Model evaluation:** Use of accuracy, precision, recall and F1-score for performance assessment.
- **Hyperparameter tuning:**  HalvingSearchCV and GridSearchCV to optimize models.

### Project Workflow:
1. Data exploration and visualization
2. Preprocessing: feature generation and selection, handling imbalanced data
3. Model training and testing
4. Conclusion 

### 1. Exploratory Data Analysis

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
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/5.png" width="400" height="300"/>
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

- Transactions for category grocery_pos starting from some amount are fraudulent.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/11.png" width="460" height="300"/>
</p>

- Transactions of certain amounts are fraudulent among some age groups.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/12.png" width="460" height="300"/>
</p>

- In some states fraud transactions are made among holders of cards by specific issuers' industry codes.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/13.png" width="460" height="300"/>
</p>

- Depending on population, there's higher probability of fraud based on card issuer's industry code.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/14.png" width="460" height="300"/>
</p>

### 2. Preprocessing

Based on exploratory data analysis, initial dataset has been changed:
- New features created: day, weekday, hour, industry_code and age.
- Not useful features were deleted: time, first, last, street, trans_num, unix_time, cc_num, dob, lat, long, merch_lat, merch_long, zip, city, job, merchant.
- Outliers were corrected by imputation with arbitrary value.

Resulting data set contained numeric and categorical features, which were treated separately. Categorical features were processed by MCA with 68 components resulting in approx. 100% variability saved. Numerical features were used for polynomial features generation with different degrees and tested for performance with Lasso regression classifier. Best permorming set had degree 2. Top 60 most important features were selected using Lasso regression.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/20.png" width="400" height="300"/>
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/21.png" width="400" height="300"/>
</p> 

To improve target class balance in the dataset, several sampling algorithms were tested by different models. Unfortunately no performance gain was achieved by any resampling method. Also, overfitting on test data was introduced by all resamplers, except EditedNearestNeighbours.<br/>
Best scores were achieved on EditedNearestNeighbours processed data and data without resampling. Best performing models: Decision Tree, Nearest Neighbour, MLPC and Random Forest.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/22.png"/>
</p>

Models showed slightly better results on data without resampling, so no resampling method was used further.<br/>
Data preparation pipeline including all steps for data preprocessing was created as a result of this section.

### 3. Model training and testing

For best performing models from previous step hyperparameter tuning was performed.<br/>
The dataset was split into training and testing sets, with 75% of the data used for training.<br/>
For initial detection of promising hyperparameters space HalvingGridSearchCV was used. Then GridSearchCV was used to select best hyperparameters from the area. F1 score was used for scoring.<br/>
Repeated Stratified K-Fold cross validation was used to validate performance and avoid overfitting.<br/><br/>

KNeighbors classifier didn't show noticeable increase of F1 score after hyperparameters tuning. Combined with long training time it makes this model unsuitable for this project.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/30.png" width="460" height="300"/>
</p>

Decision Tree model showed minor improvement in comparison to baseline results. 

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/31.png" width="460" height="300"/>
</p>

Random Forest showed promising results.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/32.png" width="460" height="300"/>
</p>

Multi-layer Perceptron classifier showed the best results with 0.8455 F1 score and 0.9983 accuracy on test data.<br/>
After hyperparameters tuning, the grand test on 100% previously unused data was performed. Trained model achieved 0.8408 F1 score and 0.9982 accuracy score, indicating its suitability for further usage.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/33.png" width="460" height="300"/>
</p>

A pipeline was developed, containing all data preprocessing steps and the estimator with highest score.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/34.png" width="353" height="409"/>
</p>

### 4. Conclusion 

This project successfully demonstrates the application of machine learning techniques to address the challenge of credit card fraud detection. By employing rigorous exploratory data analysis, insightful patterns and anomalies were uncovered, contributing to a better understanding of fraud characteristics. Interestingly, the models performed better on unbalanced dataset, underscoring the importance of cautious resampling usage. A robust preprocessing pipeline was developed, incorporating feature engineering and advanced dimensionality reduction techniques.<br/>
Through systematic evaluation and hyperparameter tuning, the Multi-Layer Perceptron Classifier emerged as the best-performing model, achieving an F1 score of 0.8408 and an accuracy of 99.82% on previously unseen test data. These results underline the model’s reliability and potential for real-time fraud detection in practical applications.