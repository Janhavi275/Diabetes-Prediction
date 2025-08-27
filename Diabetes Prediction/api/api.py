from flask import Flask, request, jsonify
import numpy as np
from model import train_model, predict_diabetes  # Import model functions

app = Flask(__name__)

# Train model and load it
data_file = "PimaIndiansDiabetes.csv"  # Update with your dataset path
model, scaler, accuracy = train_model(data_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array([
            float(data['Pregnancies']),
            float(data['Glucose']),
            float(data['BloodPressure']),
            float(data['SkinThickness']),
            float(data['Serum']),
            float(data['BMI']),
            float(data['DiabetesPedigreeFunction']),
            float(data['Age'])
        ])

        # Get prediction
        prediction = predict_diabetes(model, scaler, input_data)

        return jsonify({
            'prediction': int(prediction),
            'model_accuracy': round(accuracy, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on port 5001
