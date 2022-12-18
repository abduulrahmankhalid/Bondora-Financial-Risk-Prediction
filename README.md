# **Bondora-Financial-Risk-Prediction**

### Credit risk modeling of peer-to-peer lending Bondora systems. Applying Data Cleaning to Bondora Raw Loans Dataset, Exploratory Data Analysis and Visualizations for the data. Feature Engineering for the Features Set. Classifications and Regression Modeling with Pipelines, and Finally Deployment to a Web Application using Flask and Render.

#### This Repository is Part of the Machine Learning Engineer Internship at Technocolabs

### Deployed Web App at Render : https://bondora-financial-risk-predictor.onrender.com/

# **Abstract**
- In this project we will be doing credit risk modelling of peer to peer lending Bondora systems, Peer-to-peer lending has attracted considerable attention in recent years, largely because it offers a novel way of connecting borrowers and lenders. But as with other innovative approaches to doing business, there is more to it than that. Some might wonder, for example, what makes peer-to-peer lending so different–or, perhaps, so much better–than working with a bank, or why has it become popular in many parts of the world.

- Certainly, the industry has witnessed strong growth in recent years. According to Business Insider, transaction volumes in the U.S. and Europe, the world’s leading P2P markets, have expanded at double and, in some cases, triple-digit percentage rates, bolstered by widespread acceptance of doing business online and a supportive regulatory environment.

- For investors, "peer-2-peer lending," or "P2P," offers an attractive way to diversify portfolios and enhance long-term performance. When they invest through a peer-to-peer platform, they can profit from an asset class that has proven itself in both good times and bad. Equally important, they can avoid the risks associated with putting all their eggs in one basket, especially at a time when many experts believe that traditional favorites such as stocks and bonds are riskier than ever.

- Default risk has long been a significant risk factor to test borrowers’ behaviour in Peer-to-Peer (P2P) lending. In P2P lending, loans are typically uncollateralized and lenders seek higher returns as compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision and realize the return that compensates for the risk.

- As in the financial research domain, there are very few datasets available that can be utilized for building and analyzing credit risk models. This dataset will help the research community in building and performing research in the credit risk domain.


# **Understanding the Data**
- Data for the study has been taken from a leading European P2P lending platform (Bondora).The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between 1st March 2009 and 27th January 2020. The data comprises of demographic and financial information of borrowers, and loan transactions.In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk.

- Dataset Attributes Definitions Can be Found [Here](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Attributes_Definitions.ipynb)

# **Intial Data Preprocessing**
- From the first look at the data we knew that we have a huge dataset with (134529, 112) rows and columns , huge number of the rows had null values, so we decided to drop first features that had more than 40% of null values, decreasing the number of features to 77 feature, then drop some features which will have no role in default prediction , so we managed to get the data to have 48 relevant features,

- After some cleaning we moved to  **Creating Target Variable** for the Classification of Loan Stautus

  - Here, status is the variable which help us in creating target variable. The reason for not making status as target variable is that it has three unique values **current, Late and repaid**. There is no default feature but there is a feature **default date** which tells us when the borrower has defaulted means on which date the borrower defaulted. So, we will be combining **Status** and **Default date** features for creating target variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null.
So we will first filter out all the current status records because they are not matured yet they are current loans. 

- Now with checking the datatypes for features and search for any data type mismatch
  - First we will delete all the features related to date as it is not a time series analysis so these features will not help in predicting target variable.
  -  there are many columns which are present as numeric but they are actually categorical as per data description So we will convert these features to categorical features with proper mapping to their categories.
   
- Finally after preprcoessing and cleaning the data we can continue with EDA.

- **You Can refert to these steps in the [Preprocessing Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Bondora_Complete_Preprocessing.ipynb)**

# **Exploratory Data Analysis**
- First of all we need to perform data cleaning and fill all the null values , we seperated the categorical features and filled the null values in it with the Mode, and for the Numerical Features we fill it with the Mean. and dropped unwanted strings columns.

![nulls](https://user-images.githubusercontent.com/76521677/208266819-27c131f6-5c2c-45db-af5b-38bc1b82be91.png)

> Visualizing the missing values

- Now we ready for visualizing the data, Most of the Features were highly postively right skewed as we can see here

![skewed features](https://user-images.githubusercontent.com/76521677/208266139-6604abad-1d4f-49c8-ab85-d538ed84d29a.png)

- and the categorical features had most of it's categories as Not Set/Not Specified like some of these features

![notset](https://user-images.githubusercontent.com/76521677/208266181-976bc06e-5210-4e47-a348-55200f60fcab.png)

- Also most of the features suffering from high percentage of outliers as per these features

![outliers](https://user-images.githubusercontent.com/76521677/208266214-e002da68-d197-4c21-85b3-6d1df507b1bf.png)

- But we had intersting insights about the data we can found here, like most of the borrowers are males whome martial status in not specified, and most of them are from estonia with secondary eduction not specifying the use of the laon.

![insights](https://user-images.githubusercontent.com/76521677/208266267-cde73feb-2dba-4b3f-813d-5b5f9771d1d2.png)

- Let's have a look at the insights between features and the targer variable, as expected most of the accepted loans are for males but with more acceptence rate from spain and finland for secondary and vocational education borrowors

![targetcats](https://user-images.githubusercontent.com/76521677/208266539-85b7f368-8aeb-434d-90ef-2af241f77e9b.png)

- Also intersting inights that most of the accepted loans are for higher age borrowores and most of the loans are in a small range up to 4000.

![age](https://user-images.githubusercontent.com/76521677/208266733-c2c48215-4af9-4c92-a6e2-5c0d2f8496a7.png)


- Now, Let's have a look at the correlation between features and the target variable

![target](https://user-images.githubusercontent.com/76521677/208266425-1fef37b9-2686-4bc5-8f80-8bd2d345b088.png)

> Looks like our Target Variable have a strong linear relationship with `PrincipalBalance` which is highly postive correlation , and highly negative correlation with `PrincipalPaymentsMade`, Also we can say low positive correlation `InterestAndPenaltyBalance`   

- And a look at the corraltion matrix for the whole data 

![corr](https://user-images.githubusercontent.com/76521677/208266756-ab449a9f-8068-44e8-a24c-19ceb5dd2c36.png)

> Looks like these features are highly positive correlated with each other

> `AppliedAmount , Amount , MonthlyPayment , PrincipalBalance` 

> `ExistingLiabilities , NoOfPreviousLoansBeforeLoan , AmountOfPreviousLoansBeforeLoan`

> `PrincipalPaymentsMade , BidsPortfolioManager , InterestAndPenaltyPaymentsMade`

> Also Looks like NewCreditCustomer Columns are high negatively corralted with these features

> `NewCreditCustomer` ----> `AmountOfPreviousLoansBeforeLoan, ExistingLiabilities, NoOfPreviousLoansBeforeLoan`

- **You Can See Far more Insights in the [EDA Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Bondora_Complete_EDA.ipynb)**

# **Feature Engineering**

- Starting with the Mutual information selection between features and target variable , To detect any kind of relationship, Unlike correlation that can only detect linear relationships.

![mi](https://user-images.githubusercontent.com/76521677/208267275-4801be69-b185-4085-9305-9ce5760fc99c.png)

> We can that the top 3 features are the ones that have high corralation with the target variables.

- Let's fast check the model performence to make a baseline before feature engineering

![test](https://user-images.githubusercontent.com/76521677/208267335-f3aaf8ae-4923-438a-87f1-6a6e7e912f63.png)

> and here we get 99% accuaracy without any feature engineering using random forest classifer, obviously it is overfitting the top 3 features we have which are leaky features that are made after the loan has been accepted, and due to they won't be available at deployment , we will drop those 3 features

- After dropping the leaky features we get reasonalbe intial accuarcay with 74%, with proper feature importance to most features

![test2](https://user-images.githubusercontent.com/76521677/208267434-afdf3c34-6de9-4ad0-9f65-67f95e87fc79.png)

- Moving on , we will remove outliers with percetiles only between `quantile([0.001, 0.99]` , resulting in decreasing number of rows from `77393` to `67314 `.

- Then Encoding Categorcial Features with Label Encoding using pandas `factorize` method.

- And Normalizing the features with sklearn `StandardScaler` and transforming the features making the highly skewed features less skewed with `PowerTransformer`.

- Jumping on to Cluster analysis, First by finding the optimal number of clusters with the elbow method

![clusters](https://user-images.githubusercontent.com/76521677/208267630-4563e618-7c46-4f8a-96fb-561fce846e95.png)

> Clustering was't very helpful with increasing accuracy

- Now with PCA
  - Let's first Find Optimum Number of Components with the elbow method
   
     ![pca](https://user-images.githubusercontent.com/76521677/208267673-94d36cf2-e5fa-4d9d-910f-1c35221066d0.png)
     
  - And Testing the Performence with the selected 16 components of PCA, Was't very great. Only 71% Acc
  
    ![pca test](https://user-images.githubusercontent.com/76521677/208267739-d7c9dd5c-d251-42a6-be0f-b791d3662cfa.png)
    
- Let's now perform Feature Selection 
  - First Dropping Most Corralted Features to each other
 
  - Seconly using RFE (Recursive Feature Elimination) with Random Forest, Selecting the top 20 most important features
  
  ![rfe](https://user-images.githubusercontent.com/76521677/208267821-246afc70-f815-497e-865e-8cdace347a9e.png)
  
  > Nearly no diffrence in Accuracy with 74% also, but with only 20 features not 47.

- Finally, Comparing Performence for Selected Features with XGboost and Logistic Regression
  
  - XGBoost intersingly gave less accuracy than Random Forest 71%

  - Not so different with Logistic Regression giving about 70% Accuracy.

- **You Can refert to these steps in the [Feature Engineering Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Feature_Engineering_Team_A.ipynb)**

# **Classification Modeling**
- We know that our current accuracy is 74% with random forest using 20 features after feature engineering.
- We will use two models in the classification part `RandomForestClassifier` and `LogisticRegression`.
- We will first make a pipleine that has the preprocessing steps `StandardScaler` and `PowerTransformer` with the random forest model and other with the logostic regression.
- Now, With the **Hyper Parameters Tuning**
  - Firstly trying grid search with 3 diffrent values for Random Forest `n_estimators, max_features, max_depth, min_samples_split , min_samples_leaf` , and `penalty, C, Solver` for Logitic Regression , Resulting in not very increase in accuracy with Random Forest and in a slight increase 70% with Logistic Regression.
  - Secondly with Randomized Search with more range in values for the same hyper parameters , resulting in a slight increase in accuracy 75% with Random Forest. But same accuracy with Logistic Regression.
  
- Let's Proceed with Model Evaluation
  - with Random Forest Roc Auc Score was 80% 
 
    ![rf](https://user-images.githubusercontent.com/76521677/208268956-59c92676-a7ff-4767-b472-58281d1dd3b2.png)
    
    ![rf1](https://user-images.githubusercontent.com/76521677/208268970-d866a15b-9e06-4255-ad61-3433c43bfd8c.png)
    
    > with Classification Matrix for Random Forest Model We can see the Model Struggling with the Not Defualted Class.
   
   - with Logistic Regression Roc Auc Score was only 70% 
 
    ![lr](https://user-images.githubusercontent.com/76521677/208269026-546a1607-c9a2-409e-b13b-3f4f92c0aa03.png)
    
    ![lr1](https://user-images.githubusercontent.com/76521677/208269029-fc344386-5efb-465a-b8b8-04e650591f13.png)

    > with Classification Matrix for Logistic Regression  We can see the Model also Struggling with the Not Defualted Class.

- We will proceed with the Random Forest Model due to it's better accuracy (75%).

- **You Can refert to these steps in the [Classification Modeling Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Classification_Modeling_Team_A.ipynb)**

# **Regression Modeling**
- for the Regression Part we have three Target Varialbes
  - Preferred EMI (Monthly Payment)
  - Repayment Years (Should be able to pay the loan for that period)
  - ROI (Return on Investment)
 
- We will first Create these Features
  - EMI = [P x R x (1+R)^N]/[(1+R)^N-1], where P stands for the loan amount or principal, R is the interest rate per month [if the interest rate per annum is 11%, then the rate of interest will be 11/(12 x 100)], and N is the number of monthly instalments. Using `Amount, Interest, LoanDuration` Features
   - ROI = Investment Gain / Investment Base , ROI = Amount lended * interest/100
  > ROI = Interest Amount / Total Amount. Creating First `InterestAmount , TotalAmount`. 
  - Repayment Years : Calculting how many months to pay the full total amount of the loans , then dividing it to years.

- After Creating the 3 Target Variables we will try `Linear Regression` and `Adaboost Regressor` Models
- with `Linear Regression` the Accuracy was not that bad.
  - giving 66% for `Repayment Years` 
  - Interstingly giving 86% for `EMI`
  - giving 95% for `ROI`

- with `Adaboost Regressor` the Accuracy was far better.
  - giving 86% for `Repayment Years`  
  - Interstingly giving 84% for `EMI`
  - giving 99% for `ROI`

- We will proceed with the `Adaboost Regressor` due to it's better accuracy in the 3 Target Variables.

- **You Can refert to these steps in the [Regression Modeling Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Regression_Modeling_Team_A.ipynb)**

# **Making Piplines**

- First we will save the trained models in a pickle files to load them fast when we make the web application with Flask

- For the Classification Model `Random Forest` Model Size was above 100MB , So we had to compress it with Joblib Library, Making it about 20MB.

- and for Regression Model we saved the three `Adaboost Regressor` Models trained on sepeate targer variables.

- Also Saving the Preprocessing Pipeline to Transform new Prediction Data. 

- Then We Proceed to make one prediction function that takes as an input the 20 features we selected in the Feature Engineering Part.

- The Prediction Function had
  
  - Process input data , handel their datatypes to same as original trained data.
  - Encoding Catergorical Features the same way Encoded in training
  - Transformin the data with the Preprcoessing Pipeline, to be in the same format the Models trained on.
  - Finally Making Predictions with the saved models.
  - Outputing a dataframe that has the full four target variables    


- Finally we Checked accuracy predicting the Whole Data
  - For the Classification Modeling , the Accuracy was pretty good about 94%.
  - For the Regression Modeling, it had a Slighly Inrease in Some of the Models. 

- **You Can refert to these steps in the [Piplines Notebook](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/blob/main/Regression_Modeling_Team_A.ipynb)**

# **Deployment**

- We Began with Making the Front End to the Web Application, Then Implementing the Prediction Function and making sure to pass the required inputs , and it's returning the proper outputs.

- After Making Sure the Web app was Running Correctly , Giving Same Results we had Befor. We Proceed with Deployment

- We Deployed the Web App with **Render.com** , After Installing the Required Dependencies on the Cloud Server.

- You Can Check the Deployed Web App [Here](https://bondora-financial-risk-predictor.onrender.com/).

- **You Can refert to these steps in the [Deployment Branch](https://github.com/abduulrahmankhalid/Bondora-Financial-Risk-Prediction/tree/Deployment).**
