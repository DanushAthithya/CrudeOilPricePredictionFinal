from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
from datetime import datetime, timedelta

app = Flask(__name__,template_folder='templates')
model=pickle.load(open('arima.pkl','rb'))


exogg = pd.read_csv('ndata.csv', index_col='Date', parse_dates=True)

def rectify(arr):
    arr1=arr.split("   ")
    print(arr1)
    return arr

@app.route('/')
def main():
    return render_template('index.html')
# Define route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input for start and end dates
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # Calculate the difference between start and end date
        date_diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        end_date_d=pd.to_datetime(end_date)
        # Use the loaded model and data to make predictions for the specified dates
        prediction = model.predict(start=start_date, end=end_date, 
                                                 exog=exogg[-date_diff-1:])
        predic=""
        for i in range(len(prediction)):
            d=end_date_d.date()
            predic=predic+"\n"+str(d)+"   "+str(prediction.get(i))
            end_date_d -= timedelta(days=1)
            
        # Render the prediction result in a template
        return render_template('index.html', prediction=predic)
  

if __name__ == '__main__':
    app.run(debug=True)
