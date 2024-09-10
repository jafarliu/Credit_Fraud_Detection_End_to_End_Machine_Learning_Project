import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor (scaler)
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))  # Load the preprocessor pipeline

# Helper function to map output to human-readable labels
def map_prediction_label(prediction):
    return "No Fraud" if prediction == 0 else "Fraud"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api(): 
    data = request.json['data']  
    print(data)
    
    # Transform the input data using the preprocessor
    transformed_data = preprocessor.transform(pd.DataFrame([data]))  # Transforming as a DataFrame
    
    # Make the prediction
    prediction = model.predict(transformed_data)[0]  # Get the first prediction
    
    # Map the prediction to a human-readable label
    output_label = map_prediction_label(prediction)
    print(output_label)
    
    return jsonify(output_label)

@app.route('/predict', methods=['POST'])
def predict():
    data = []
    
    # Collect form input data and convert to float (for numerical inputs)
    for value in request.form.values():
        try:
            data.append(float(value))  # Handle numeric inputs
        except ValueError:
            data.append(value)  # If not numeric, it could be categorical
    
    # Convert data into a DataFrame for the preprocessor
    data_df = pd.DataFrame([data], columns=request.form.keys())
    
    # Transform the input data using the preprocessor
    transformed_data = preprocessor.transform(data_df)
    
    # Make the prediction
    prediction = model.predict(transformed_data)[0]  # Get the first prediction
    
    # Map the prediction to a human-readable label
    output_label = map_prediction_label(prediction)
    
    return render_template("home.html", prediction_text="The transaction  is: {}".format(output_label))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
