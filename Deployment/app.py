import pandas as pd
from flask import Flask, render_template, url_for, request,Markup,send_file

from models import Make_Predictions

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    Inputs = {}

    Inputs['BidsPortfolioManager'] = request.form.get('BidsPortfolioManager')
    Inputs['BidsApi'] = request.form.get('BidsApi')
    Inputs['BidsManual'] = request.form.get('BidsManual')
    Inputs['LanguageCode'] = request.form.get('LanguageCode')
    Inputs['Age'] = request.form.get('Age')
    Inputs['Interest'] = float(request.form.get('Interest')) / 100
    Inputs['LoanDuration'] = request.form.get('LoanDuration')
    Inputs['Education'] = request.form.get('Education')
    Inputs['MaritalStatus'] = request.form.get('MaritalStatus')
    Inputs['EmploymentDurationCurrentEmployer'] = request.form.get('EmploymentDuration')
    Inputs['HomeOwnershipType'] = request.form.get('HomeOwnershipType')
    Inputs['IncomeTotal'] = request.form.get('TotalIncome')
    Inputs['ExistingLiabilities'] = request.form.get('ExistingLiabilities')
    Inputs['LiabilitiesTotal'] = request.form.get('TotalLiabilities')
    Inputs['DebtToIncome'] = request.form.get('DebtToIncome')
    Inputs['FreeCash'] = request.form.get('FreeCash')
    Inputs['MonthlyPaymentDay'] = request.form.get('MonthlyPaymentDay')
    Inputs['Rating'] = request.form.get('Rating')
    Inputs['PreviousRepaymentsBeforeLoan'] = request.form.get('PreviousRepaymentsBeforeLoan')
    Inputs['Amount'] = request.form.get('Amount')

    Inputs = pd.DataFrame.from_dict(Inputs, orient='index').T

    Outputs = Make_Predictions(Inputs)

    Defaulted = Outputs['Defaulted'].values[0]
    ROI = Outputs['ROI'].values[0]
    EMI = Outputs['EMI'].values[0]
    RepaymentYears = Outputs['RepaymentYears'].values[0]


    Report = pd.concat([Inputs, Outputs], axis=1)

    Report.to_csv("Report.csv", index=False)

    

    if Defaulted == 1:
        return render_template('home.html', prediction_text=Markup(
            f"<br/>&#8226; Congratualtion The Loan Should Be Defaulted. <br/><br/> \
             &#8226; Your EMI (Preferred Monthly Payment): {EMI:.2f}€ <br/><br/> \
             &#8226; Your ROI (Preferred Return on Investment): {ROI:.2f}% <br/><br/> \
             &#8226; Your Repayment Years (Should be Able to Pay the Loan for): {RepaymentYears:.2f} Years ({round(RepaymentYears*12)} Months)")) 
    else:
        return render_template('home.html', prediction_text=Markup( 
            f"<br/>&#8226; Unfortunately the Loan Shouldn't Be Defaulted. <br/><br/>\
             &#8226; Your EMI Should have been (Preferred Monthly Payment): {EMI:.2f}€ <br/><br/> \
             &#8226; Your ROI Should have been (Preferred Return on Investment): {ROI:.2f}% <br/><br/> \
             &#8226; Your Repayment Years Should have been (Should have been Able to Pay the Loan for): {RepaymentYears:.2f} Years ({round(RepaymentYears*12)} Months)")) 


@app.route('/download', methods=['POST'])
def download():
    return send_file("Report.csv", download_name="Loan Report.csv")

if __name__=="__main__":
	app.run(debug=True)