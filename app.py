from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'best_solar_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Solar Energy Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected JSON format:
    {
        "hour": 12,
        "day_of_week": 3,
        "month": 6,
        "day_of_month": 15,
        "power_prev_1h": 150.5,
        "power_prev_2h": 148.2,
        "power_prev_4h": 145.0,
        "power_mean_24h": 155.0,
        "power_std_24h": 25.5
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['hour', 'day_of_week', 'month', 'day_of_month', 
                          'power_prev_1h', 'power_prev_2h', 'power_prev_4h',
                          'power_mean_24h', 'power_std_24h']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400
        
        # Prepare features in correct order
        features = np.array([[
            data['hour'],
            data['day_of_week'],
            data['month'],
            data['day_of_month'],
            data['power_prev_1h'],
            data['power_prev_2h'],
            data['power_prev_4h'],
            data['power_mean_24h'],
            data['power_std_24h']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'unit': 'MW',
            'input_features': data
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    Expected JSON format: list of feature dictionaries
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected a list of predictions'}), 400
        
        results = []
        for idx, item in enumerate(data):
            required_fields = ['hour', 'day_of_week', 'month', 'day_of_month', 
                              'power_prev_1h', 'power_prev_2h', 'power_prev_4h',
                              'power_mean_24h', 'power_std_24h']
            
            if not all(field in item for field in required_fields):
                results.append({'index': idx, 'error': 'Missing required fields'})
                continue
            
            features = np.array([[
                item['hour'],
                item['day_of_week'],
                item['month'],
                item['day_of_month'],
                item['power_prev_1h'],
                item['power_prev_2h'],
                item['power_prev_4h'],
                item['power_mean_24h'],
                item['power_std_24h']
            ]])
            
            prediction = model.predict(features)[0]
            results.append({
                'index': idx,
                'prediction': float(prediction),
                'unit': 'MW'
            })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
