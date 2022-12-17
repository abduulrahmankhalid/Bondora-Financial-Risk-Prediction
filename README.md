# **Bondora-Financial-Risk-Prediction**

#### This Repository is Part of the Machine Learning Engineer Internship at Technocolabs

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
