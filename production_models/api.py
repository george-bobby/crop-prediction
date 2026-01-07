
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load predictor
predictor = joblib.load("production_models/crop_predictor.pkl")

@app.route('/predict', methods=['POST'])
def predict_yield():
    """API endpoint for crop yield prediction."""
    try:
        data = request.json

        # Validate required fields
        required = ['crop', 'N', 'P', 'K', 'rainfall', 'area']
        for field in required:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400

        # Make prediction
        result = predictor.predict(
            crop=data['crop'],
            N=data['N'],
            P=data['P'],
            K=data['K'],
            rainfall=data['rainfall'],
            area=data['area']
        )

        # Add request info to response
        result['request'] = {
            'crop': data['crop'],
            'timestamp': pd.Timestamp.now().isoformat()
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'timestamp': pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
