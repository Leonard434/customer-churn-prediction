from flask import Flask, jsonify, request, render_template
import sys, os
import logging
import pandas as pd
logging.getLogger(__name__).setLevel(logging.INFO)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_single_customer, predict_batch

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''Make a prediction for a single customer.'''
    logging.info("Received prediction request.")
    try:
        data = request.get_json()
        if data is None:
            logging.error("No JSON data received.")
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        result = predict_single_customer(data)
    except Exception as e:
        logging.error(f"Error occurred while making prediction: {e}")
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while making the prediction'}), 500
    logging.info("Prediction completed.")
    return jsonify({'status': 'success', 'prediction': result}), 200

@app.route('/predict_batch', methods=['POST'])
def predict_batch_route():
    '''Make predictions for a batch of customers.'''
    logging.info("Received batch prediction request.")
    try:
        data = request.get_json()
        if data is None or 'customers' not in data:
            logging.error("No customer data received.")
            return jsonify({'status': 'error', 'message': 'No customer data provided'}), 400
        customers_df = pd.DataFrame(data['customers'])
        results_df = predict_batch(customers_df)
        results = results_df.to_dict(orient='records')
    except Exception as e:
        logging.error(f"Error occurred while making batch predictions: {e}")
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while making batch predictions'}), 500
    logging.info("Batch prediction completed.")
    return jsonify({'status': 'success', 'predictions': results}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)