from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

model = joblib.load('crop_recommendation_model.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    N = float(data['N'])
    P = float(data['P'])
    K = float(data['K'])
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    rainfall = float(data['rainfall'])

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)

    return jsonify({'crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
