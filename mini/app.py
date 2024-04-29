from flask import Flask, render_template, request
from models.diabetes_model import DiabetesModel

app = Flask(__name__, static_url_path='/static')
model = DiabetesModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET'])
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    pregnancies = int(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = int(request.form['age'])

    user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    prediction = model.predict(user_input)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)