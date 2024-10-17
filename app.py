import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Import SVC model and standard scaler pickle
svc_model = pickle.load(open('models/svc.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    result_text = None
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        # Scaling the input data
        new_data_scaled = standard_scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = svc_model.predict(new_data_scaled)[0]  # Get the single prediction value

        # Set result_text based on the prediction
        if result == 0:
            result_text = 'The person is not diabetic.'
        else:
            result_text = 'The person is diabetic.'

        # Pass the result_text to the template
        return render_template('index.html', results=result_text)
    
    return render_template('index.html', results=result_text)

if __name__ == "__main__":
    app.run(debug=True)
