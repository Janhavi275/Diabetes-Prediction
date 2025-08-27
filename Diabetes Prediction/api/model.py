import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data_file):
    data_set = pd.read_csv(data_file)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model trained with accuracy: {accuracy:.4f}")

    return model, scaler, accuracy

def get_user_input():

    times_pregnant = float(input("Enter number of times pregnant: "))
    glucose_concentration = float(input("Enter glucose concentration: "))
    blood_pressure = float(input("Enter blood pressure: "))
    skin_thickness = float(input("Enter skin thickness: "))
    serum = float(input("Enter serum level: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree = float(input("Enter diabetes pedigree function: "))
    age = float(input("Enter age: "))

    input_data = [times_pregnant, glucose_concentration, blood_pressure,
                  skin_thickness, serum, bmi, diabetes_pedigree, age]
    return input_data

def predict_diabetes(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])

    prediction = model.predict(input_data_scaled)[0]
    return prediction


data_file = 'PimaIndiansDiabetes.csv' 
model, scaler,  accuracy = train_model(data_file)
user_input = get_user_input()
prediction = predict_diabetes(model, scaler, user_input)

if prediction == 1:
    print("Prediction: Diabetes")
else:
    print("Prediction: No Diabetes")

print(f"Model Accuracy: {accuracy:.4f}")
