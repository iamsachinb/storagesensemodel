from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Reconstruct the Keras model from the configuration
model = tf.keras.models.model_from_json(model_data['config'])
model.set_weights(model_data['weights'])

# Recreate the scaler with the same feature range as during training
scaler = MinMaxScaler(feature_range=(0, 1))  # Adjust feature_range if different

# Fit the scaler with hypothetical min and max values used during training
# You'll need to manually specify these based on the training data
scaler.fit(np.array([
    [10, 20, 500, 10, 0],  # hypothetical min values of your input features
    [100, 200, 10000, 100, 300]  # hypothetical max values of your input features
]))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        required_keys = ['airTemperature', 'processTemperature', 'rotationalSpeed', 'torque', 'toolWear']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing required input: {key}'}), 400

        # Prepare input features
        input_features = np.array([[data['airTemperature'], data['processTemperature'], data['rotationalSpeed'], data['torque'], data['toolWear']]])
        logging.info(f"Received input features: {input_features}")

        # Scale the input features using the recreated scaler
        scaled_features = scaler.transform(input_features)
        logging.info(f"Scaled features: {scaled_features}")

        # Reshape the input features for the model
        input_features_reshaped = np.repeat(scaled_features, 10, axis=0).reshape(1, 10, 5)

        # Get the prediction
        prediction = model.predict(input_features_reshaped)
        maintenance_time = prediction[0][0].tolist()  # Convert numpy array to list for JSON serialization

        maintenance_time = maintenance_time * 100

        return jsonify({'maintenanceTime': maintenance_time})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
