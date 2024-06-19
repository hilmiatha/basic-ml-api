import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger

with open('./model_basic_randomforest.pkl', 'rb') as model_basic_randomforest_pkl:
    model = pickle.load(model_basic_randomforest_pkl)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict_iris():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: s_length
        in: formData
        type: number
        required: true
      - name: s_width
        in: formData
        type: number
        required: true
      - name: p_length
        in: formData
        type: number
        required: true
      - name: p_width
        in: formData
        type: number
        required: true
    responses:
      200:
        description: Prediction
        schema:
          type: string
    """
    s_length = float(request.form.get('s_length'))
    s_width = float(request.form.get('s_width'))
    p_length = float(request.form.get('p_length'))
    p_width = float(request.form.get('p_width'))
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_iris_with_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Prediction
        schema:
          type: array
          items:
            type: string
    """
    x = pd.read_csv(request.files.get('input_file'), header=None)
    prediction = model.predict(x)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
