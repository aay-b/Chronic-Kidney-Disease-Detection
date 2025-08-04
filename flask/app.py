import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
app._got_first_request = False
model = pickle.load(open('CKD.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction', methods=['POST', 'GET'])
def show_prediction_page():
    return render_template('indexnew.html')
@app.route('/Home', methods=['POST', 'GET'])
def show_home_page():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Example of collecting all 25 fields (you must collect ALL of them from the form)
    id = float(request.form['id'])
    age = float(request.form['age'])
    blood_pressure = float(request.form['blood_pressure'])
    specific_gravity = float(request.form['specific_gravity'])
    albumin = float(request.form['albumin'])
    sugar = float(request.form['sugar'])
    red_blood_cells = 1 if request.form['red_blood_cells'] == 'abnormal' else 0
    pus_cell = 1 if request.form['pus_cell'] == 'abnormal' else 0
    pus_cell_clumps = int(request.form['pus_cell_clumps'])
    bacteria = int(request.form['bacteria'])
    blood_glucose_random = float(request.form['blood_glucose_random'])
    blood_urea = float(request.form['blood_urea'])
    serum_creatinine = float(request.form['serum_creatinine'])
    sodium = float(request.form['sodium'])
    potassium = float(request.form['potassium'])
    haemoglobin = float(request.form['haemoglobin'])
    packed_cell_volume = float(request.form['packed_cell_volume'])
    white_blood_cell_count = float(request.form['white_blood_cell_count'])
    red_blood_cell_count = float(request.form['red_blood_cell_count'])
    hypertension = int(request.form['hypertension'])
    diabetes_mellitus = int(request.form['diabetes_mellitus'])
    coronary_artery_disease = int(request.form['coronary_artery_disease'])
    appetite = int(request.form['appetite'])
    pedal_edema = int(request.form['pedal_edema'])
    anemia = int(request.form['anemia'])

    # Match order of training data (excluding 'class' and 'id')
    input_data = np.array([[id, age, blood_pressure, specific_gravity, albumin, sugar,
                            red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                            blood_glucose_random, blood_urea, serum_creatinine, sodium,
                            potassium, haemoglobin, packed_cell_volume, white_blood_cell_count,
                            red_blood_cell_count, hypertension, diabetes_mellitus,
                            coronary_artery_disease, appetite, pedal_edema, anemia]])

    # Apply the same scaler and model
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    # Interpret result
    result = "High chance of CKD" if prediction[0] == 1 else "Low chance of CKD"
    return render_template('result.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
