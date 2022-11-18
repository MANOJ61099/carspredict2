from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Autos_Cleaned_Data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    vehicle_type = sorted(car['vehicle_type'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    month = sorted(car['month'].unique())
    fuel_type = sorted(car['fuel_type'].unique())
    gear = sorted(car['gear'].unique())
    damaged = sorted(car['damaged'].unique())

    companies.insert(0, 'Select Company')
    vehicle_type.insert(0, 'Select Vehicle Type')
    fuel_type.insert(0, 'Select Fuel Type')
    year.insert(0, 'Select Year of Reg')
    month.insert(0, 'Select Month of Reg')
    gear.insert(0, 'Select the Gear Type')
    damaged.insert(0, 'Condition of the Car')


    return render_template('index.html', companies=companies, car_models=car_models, years=year,
                           vehicle_types=vehicle_type, months=month, fuel_types=fuel_type, gears=gear, damages=damaged)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    vehicle_type = request.form.get('vehicle_type')
    year = request.form.get('year')
    month = request.form.get('month')
    gear = request.form.get('gear')
    fuel_type = request.form.get('fuel_type')
    power = request.form.get('ps')
    kms_driven = int(request.form.get('kilo_driven'))
    damaged = request.form.get('damaged')
    print(company,car_model, vehicle_type, year, month, gear, fuel_type, power, kms_driven, damaged )

    prediction = model.predict(pd.DataFrame([[company, car_model, vehicle_type, year, month, gear, fuel_type, power, kms_driven, damaged]], columns=['company', 'name', 'vehicle_type', 'year', 'month', 'gear', 'fuel_type', 'ps', 'kms_driven', 'damaged']))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()
