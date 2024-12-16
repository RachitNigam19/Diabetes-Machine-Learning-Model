# Importing all necessary dependencies
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:  # Corrected mode to 'rb' for binary reading
    model = pickle.load(file)


# Initialize the Flask app
app = Flask(__name__)

# Load the scaler used during training
scaler = StandardScaler()
scaler_path = 'scaler.pkl'
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is present in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        int_features = [float(x) for x in request.form.values()]  # Handle float inputs (e.g., BMI, DPF)
        final_features = np.array([int_features])  # Convert to 2D array

        # Standardize the input using the same scaler used during training
        standardized_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(standardized_features)
        output = 'This Patient is Diabetic' if prediction[0] == 1 else 'This Patient is NON-Diabetic'

        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
