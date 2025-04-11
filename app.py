from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Example: Ensure this matches the input format your model expects
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    result = {
        'prediction': 'Approved' if prediction == 1 else 'Rejected',
        'approval_probability': round(probabilities[1] * 100, 2),
        'default_probability': round(probabilities[0] * 100, 2)
    }
    return jsonify(result)
