from flask import Flask, render_template, request, jsonify, session
import requests

app = Flask(__name__)
API_URL = "http://127.0.0.1:5001/predict"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            data = {
                'Pregnancies': float(request.form['Pregnancies']),
                'Glucose': float(request.form['Glucose']),
                'BloodPressure': float(request.form['BloodPressure']),
                'SkinThickness': float(request.form['SkinThickness']),
                'Serum': float(request.form['Serum']),
                'BMI': float(request.form['BMI']),
                'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
                'Age': float(request.form['Age'])
            }

            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                prediction = response.json()['prediction']
            else:
                error = "Error in prediction. Try again."

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
