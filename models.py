import pandas as pd
import joblib
import pickle

def Make_Predictions(df):
  
  # Converting Categoricals
  cat_cols=['LanguageCode','Education','MaritalStatus','EmploymentDurationCurrentEmployer',
          'HomeOwnershipType','Rating']
          
  for col in cat_cols:
    df[col] = df[col].astype("category")

  # Converting Numberical
  int_cols=['BidsPortfolioManager', 'BidsApi', 'Age', 'LoanDuration',
       'ExistingLiabilities', 'MonthlyPaymentDay']
          
  for col in int_cols:
    df[col] = df[col].astype("int")


  float_cols=['BidsManual', 'Interest', 'IncomeTotal', 'LiabilitiesTotal',
       'DebtToIncome', 'FreeCash', 'PreviousRepaymentsBeforeLoan', 'Amount']
          
  for col in float_cols:
    df[col] = df[col].astype("float")


  # Encoding Categoricals
  for colname in df.select_dtypes(["object","category","bool"]):
      df[colname], _ = df[colname].factorize()


  # Preprocessing Data
  Preprocessing_Pipeline = pickle.load(open('Models/Preprocessing_Pipeline.pkl', 'rb'))

  df = Preprocessing_Pipeline.transform(df)


  Predictions = {} 

  # Making Classification Predictions
  RF_Classifier = joblib.load(open('Models/RF_Classifier.pkl', 'rb'))

  Predictions["Defaulted"] = RF_Classifier.predict(df)
 

  # Making Regression Predictions
  
    # Repayment Years Prediction
  Ada_Repay = pickle.load(open('Models/Ada_Repay.pkl', 'rb'))

  Predictions["RepaymentYears"] = Ada_Repay.predict(df)

    # Making EMI Predictions
  Ada_EMI = pickle.load(open('Models/Ada_EMI.pkl', 'rb'))

  Predictions["EMI"] = Ada_EMI.predict(df)

    # Making ROI Predictions
  Ada_ROI = pickle.load(open('Models/Ada_ROI.pkl', 'rb'))

  Predictions["ROI"] = Ada_ROI.predict(df)
  

  # Converting to DataFrame 
  Predictions = pd.DataFrame(Predictions)

  return pd.DataFrame(Predictions)