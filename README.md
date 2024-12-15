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
3. Classification model training and testing
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
For initial detection of promising hyperparameters space HalvingGridSearchCV was used. Then GridSearchCV was used to select best hyperparameters from the area.<br/>
Repeated Stratified K-Fold cross validation was used to validate performance and avoid overfitting.

<p align="center">
<img src="https://github.com/ArtemAntonov/Credit-Card-Transactions-Fraud-Detection/blob/main/img/30.png" width="460" height="300"/>
</p>

Multi-layer Perceptron classifier showed the best results with 0.8455 f1 score and 0.9983 accuracy on test data.<br/>
After hyperparameters tuning, the grand test on 100% unused previously data was performed. Trained model achieved 0.8408 f1 score and 0.9982 accuracy score, indicating its suitability for further usage.

### 4. Conclusion 

Created a project for iterative model experimentation and development,
Performed initial feature preparation and pipelining,
Trained a production-ready model,
Deployed our model to a live-hosted endpoint and a batch pipeline that automatically updates.


The Credit Card Transactions Fraud Detection project successfully addressed the critical challenge of identifying fraudulent credit card transactions through the application of machine learning techniques. Leveraging an imbalanced dataset, the project demonstrated how thoughtful preprocessing, exploratory data analysis, and model optimization can contribute to robust and reliable fraud detection systems.

Key Findings

Exploratory Data Analysis (EDA): The project revealed significant patterns and trends, such as:

Fraudulent transactions were highly concentrated in certain cities, states, and zip codes.

Older individuals (ages 50-65) were more prone to fraudulent transactions, highlighting potential demographic risks.

Specific transaction amounts and categories were disproportionately linked to fraud.

Scammers tended to act during late-night hours (23:00-4:00), emphasizing temporal patterns in fraudulent activities.

Preprocessing Enhancements: The dataset was enriched and refined through feature engineering and dimensionality reduction:

New features such as day, weekday, hour, industry code, and age were created.

Unnecessary features were removed, and numeric and categorical data were treated independently to improve modeling efficiency.

Polynomial feature generation and Lasso regression identified the top 60 features most predictive of fraud.

Target Class Balancing: Several sampling methods, including Edited Nearest Neighbors (ENN), were evaluated to address the dataset's class imbalance. ENN showed promising results without introducing overfitting, making it a preferred method for balancing data in future applications.

Model Evaluation: Models like Decision Tree, Nearest Neighbors, MLPC, and Random Forest emerged as top performers. Interestingly, the models performed slightly better on the unbalanced dataset, underscoring the importance of cautious resampling in fraud detection scenarios.

Practical Contributions

This project provided a reproducible workflow for fraud detection, including:

A pipeline for feature selection and transformation.

Clear identification of demographic and transactional risk factors.

Insights into handling imbalanced datasets without compromising model integrity.

Final Thoughts

The methodologies and results of this project underscore the potential of machine learning in enhancing financial security. By analyzing patterns of fraudulent activity and optimizing detection strategies, this project contributes to the ongoing fight against financial fraud. Future work can build upon this solid foundation to create even more effective and scalable fraud prevention systems.