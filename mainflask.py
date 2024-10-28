from flask import Flask, render_template, request, jsonify
import subprocess
from PIL import Image
from keras.models import load_model
import numpy as np 
import os
import pickle
import torch

app = Flask(__name__)

# Define default normal values
DEFAULT_VALUES = {
    "heart_rate": 75,
    "blood_pressure_systolic": 110,
    "blood_pressure_diastolic": 80,
    "respiration_rate": 16,
    "body_temperature": 98.6,
    "blood_oxygen_level": 98,
    "glucose_level": 90,
    "fasting": 'yes',
    "total_cholesterol": 180,
    "ldl_cholesterol": 100,
    "hdl_cholesterol": 60
}

def check_heart_rate(hr):
    if 60 <= hr <= 100:
        return "Normal"
    elif hr < 60:
        return "Bradycardia (Low Heart Rate)"
    else:
        return "Tachycardia (High Heart Rate)"

def check_blood_pressure(systolic, diastolic):
    if systolic <= 120 and diastolic <= 80:
        return "Normal"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension (High Blood Pressure)"
    else:
        return "Elevated Blood Pressure"

def check_respiration_rate(rr):
    if 12 <= rr <= 20:
        return "Normal"
    elif rr < 12:
        return "Bradypnea (Low Respiration Rate)"
    else:
        return "Tachypnea (High Respiration Rate)"

def check_body_temperature(temp):
    if 97 <= temp <= 99:
        return "Normal"
    elif temp < 97:
        return "Hypothermia"
    else:
        return "Fever"

def check_blood_oxygen(spo2):
    if 95 <= spo2 <= 100:
        return "Normal"
    elif spo2 < 90:
        return "Hypoxemia (Low Blood Oxygen)"
    else:
        return "Hyperoxia (High Blood Oxygen)"

def check_glucose_level(glucose, fasting=True):
    if fasting:
        if 70 <= glucose <= 99:
            return "Normal"
        elif glucose < 70:
            return "Hypoglycemia (Low Blood Sugar)"
        else:
            return "Hyperglycemia (High Blood Sugar)"
    else:
        if glucose < 140:
            return "Normal"
        else:
            return "Hyperglycemia (High Blood Sugar)"

def check_cholesterol_level(total, ldl, hdl):
    if total <= 200 and ldl <= 100 and hdl >= 60:
        return "Normal"
    else:
        return "Hyperlipidemia (High Cholesterol)"

def get_disease_info(diseases):
    disease_info = {
        "Cardiovascular Disease": {
            "details": "Abnormal heart rate or blood pressure can lead to heart disease.",
            "prevention": [
                "Maintain a healthy diet low in saturated fats.",
                "Engage in regular physical activity.",
                "Avoid smoking and limit alcohol consumption."
            ],
            "confirmation_tests": [
                "Electrocardiogram (ECG): To check heart rhythm.",
                "Echocardiogram: To visualize heart function."
            ]
        },
        "Respiratory Disease": {
            "details": "Abnormal respiration rate can indicate respiratory distress.",
            "prevention": [
                "Avoid exposure to pollutants.",
                "Practice good hygiene to prevent infections."
            ],
            "confirmation_tests": [
                "Chest X-ray: To identify lung issues.",
                "Pulmonary function tests: To assess lung capacity."
            ]
        },
        "Infection": {
            "details": "Fever often indicates infection.",
            "prevention": [
                "Vaccination against common infections.",
                "Practice good hygiene."
            ],
            "confirmation_tests": [
                "Blood tests: To identify infection markers.",
                "Imaging studies: To locate the source of infection."
            ]
        },
        "Metabolic Disorder": {
            "details": "Abnormal blood sugar levels can lead to diabetes.",
            "prevention": [
                "Maintain a balanced diet.",
                "Regular exercise.",
                "Routine monitoring of blood sugar levels."
            ],
            "confirmation_tests": [
                "Fasting blood glucose test.",
                "Hemoglobin A1c test."
            ]
        },
        "Hyperlipidemia": {
            "details": "High cholesterol levels can lead to heart disease.",
            "prevention": [
                "Healthy diet low in trans fats.",
                "Regular physical activity.",
                "Routine cholesterol screenings."
            ],
            "confirmation_tests": [
                "Lipid panel: To measure cholesterol levels."
            ]
        }
    }

    relevant_disease_info = {disease: disease_info[disease] for disease in diseases.keys() if disease in disease_info}
    return relevant_disease_info

@app.route('/', methods=['GET', 'POST'])
def menu():
    return render_template('menu.html')

@app.route('/health_assessment', methods=['GET', 'POST'])
def health_assessment():
    if request.method == 'POST':
        # Get form data
        heart_rate = int(request.form['heart_rate'])
        blood_pressure_systolic = int(request.form['blood_pressure_systolic'])
        blood_pressure_diastolic = int(request.form['blood_pressure_diastolic'])
        respiration_rate = int(request.form['respiration_rate'])
        body_temperature = float(request.form['body_temperature'])
        blood_oxygen_level = float(request.form['blood_oxygen_level'])
        glucose_level = int(request.form['glucose_level'])
        fasting = request.form['fasting'] == 'yes'
        total_cholesterol = int(request.form['total_cholesterol'])
        ldl_cholesterol = int(request.form['ldl_cholesterol'])
        hdl_cholesterol = int(request.form['hdl_cholesterol'])

        # Checking each metric
        results = {
            "Heart Rate": check_heart_rate(heart_rate),
            "Blood Pressure": check_blood_pressure(blood_pressure_systolic, blood_pressure_diastolic),
            "Respiration Rate": check_respiration_rate(respiration_rate),
            "Body Temperature": check_body_temperature(body_temperature),
            "Blood Oxygen Level": check_blood_oxygen(blood_oxygen_level),
            "Glucose Level": check_glucose_level(glucose_level, fasting),
            "Cholesterol Levels": check_cholesterol_level(total_cholesterol, ldl_cholesterol, hdl_cholesterol)
        }

        # Combine potential diseases and combination issues into one list
        diseases = {}
                # Check for individual diseases
        if "Tachycardia" in results["Heart Rate"] or "Bradycardia" in results["Heart Rate"]:
            diseases["Cardiovascular Disease"] = "Abnormal heart rate can lead to heart failure, arrhythmias, or increased risk of stroke."
        if "Hypertension" in results["Blood Pressure"] or "Hypotension" in results["Blood Pressure"]:
            diseases["Cardiovascular Disease"] = diseases.get("Cardiovascular Disease", "") + " Abnormal blood pressure can lead to heart disease, stroke, kidney damage, and other complications."
        if "Tachypnea" in results["Respiration Rate"] or "Bradypnea" in results["Respiration Rate"]:
            diseases["Respiratory Disease"] = "Abnormal respiration rate can indicate respiratory distress, pneumonia, asthma, or other respiratory conditions."
        if "Fever" in results["Body Temperature"]:
            diseases["Infection"] = "Fever often indicates infection (like flu, COVID-19, or bacterial infections) or inflammatory conditions."
        if "Hypothermia" in results["Body Temperature"]:
            diseases["Hypothermia"] = "Can result from exposure to cold, leading to severe complications like organ failure."
        if "Hypoxemia" in results["Blood Oxygen Level"]:
            diseases["Respiratory Disease"] = diseases.get("Respiratory Disease", "") + " Low blood oxygen can lead to respiratory failure, brain damage, or organ dysfunction."
        if "Hyperglycemia" in results["Glucose Level"] or "Hypoglycemia" in results["Glucose Level"]:
            diseases["Metabolic Disorder"] = "Abnormal blood sugar levels can lead to diabetes complications, including neuropathy, retinopathy, and cardiovascular diseases."
        if "Hyperlipidemia" in results["Cholesterol Levels"]:
            diseases["Cardiovascular Disease"] = diseases.get("Cardiovascular Disease", "") + " High cholesterol levels can lead to atherosclerosis, heart attacks, and strokes."

        # Check for combination issues
        if ("Tachycardia" in results["Heart Rate"] or "Bradycardia" in results["Heart Rate"]) and "Hypertension" in results["Blood Pressure"]:
            diseases["Cardiovascular Disease"] = diseases.get("Cardiovascular Disease", "") + " Combination of abnormal heart rate and high blood pressure significantly increases cardiovascular risks."
        if "Tachypnea" in results["Respiration Rate"] and "Hypoxemia" in results["Blood Oxygen Level"]:
            diseases["Respiratory Disease"] = diseases.get("Respiratory Disease", "") + " Combination of elevated respiration rate and low blood oxygen can indicate severe conditions like COPD or ARDS."
        if "Fever" in results["Body Temperature"] and ("Tachycardia" in results["Heart Rate"] or "Bradycardia" in results["Heart Rate"]):
            diseases["Sepsis"] = "Combination of fever and abnormal heart rate can indicate potential sepsis, a life-threatening response to infection."
        if "Hyperglycemia" in results["Glucose Level"] and "Hypertension" in results["Blood Pressure"]:
            diseases["Metabolic Syndrome"] = "Combination of high glucose levels and high blood pressure increases the risk of cardiovascular disease and kidney damage."

        relevant_disease_info = get_disease_info(diseases)

        return render_template('results.html', results=results, diseases=diseases, relevant_disease_info=relevant_disease_info)

    # If the request method is GET, render the form with default values
    return render_template('index.html', default_values=DEFAULT_VALUES)

# Load the pneumonia classification model
model = load_model('./PneumoniaClassification/model/pneumonia_classifier.h5')
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Load class names
with open('./PneumoniaClassification/model/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

def classify(image):
    # Preprocess the image for the model
    image = image.resize((224, 224))  # Adjust size as needed
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_index]
    return class_names[class_index], confidence

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Open the uploaded image file
            image = Image.open(file).convert('RGB')
            # Here you would call your classification function
            class_name, conf_score = classify(image)  # Assume classify is defined elsewhere
            return jsonify({'class_name': class_name, 'confidence_score': float(conf_score)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('pnindex.html') 

# model_brain_tumor = pickle.load(open('./Brain-Tumor-Detection-master/model.pkl', 'rb'))

# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# import numpy as np
# import torch
# import torch.nn as nn

# app = Flask(__name__)

# Define the model class (Net) as before
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.fc1 = nn.Linear(128 * 26 * 26, 128)  # Update this line
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = self.pool(nn.functional.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 26 * 26)  # Update this line
#         x = nn.functional.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x

# # Load the model
# model = Net()
# model.load_state_dict(torch.load('model.pth'))  # Load the saved state
# model.eval()  # Set the model to evaluation mode

# @app.route('/brain_tumor', methods=['GET', 'POST'])
# def brain_tumor():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400

#         try:
#             # Open the uploaded image file
#             image = Image.open(file).convert('RGB')
#             image = image.resize((224, 224))  # Adjust size as needed
#             image_array = np.array(image) / 255.0  # Normalize the image
#             image_array = np.transpose(image_array, (2, 0, 1))  # Change to (C, H, W) format
#             image_tensor = torch.tensor(image_array, dtype=torch.float32)  # Convert to tensor
#             image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

#             # Make prediction
#             with torch.no_grad():
#                 predictions = model(image_tensor)
#                 predicted_class = torch.round(predictions).item()  # Get the predicted class (0 or 1)
#                 confidence = predictions.item()  # Get the confidence score

#             class_names = ['no_tumor', 'tumor']  # Adjust this based on your model's output
#             return jsonify({'class_name': class_names[int(predicted_class)], 'confidence_score': float(confidence)})
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

#     return render_template('brain_tumor.html')

# @app.route('/test_page', methods=['GET','POST'])
# def test_page():
#     return render_template('test_page.html')



if __name__ == '__main__':
    app.run(debug=True)