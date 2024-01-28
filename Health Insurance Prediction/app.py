from flask import Flask, request, jsonify
import pandas as pd
from modules.insurance_model import InsuranceModel

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Insurance API Modelling Prefiction Insurance!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    result_predict = InsuranceModel().runModel(df, typed='single')
    return jsonify({'status': 'predicted', 'predicted_result': result_predict})

if __name__ == "__main__":
    app.run(port=8000)