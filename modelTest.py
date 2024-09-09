import pandas as pd
from joblib import load

# Load the trained logistic regression model (including the pipeline)
try:
    lr_model = load('heartdisease_logisticregression.joblib')
    st.write('Model loaded successfully.')
except Exception as e:
    st.write(f"Error loading model: {e}")

# Define a sample input matching the structure of the training DataFrame
input_data = pd.DataFrame({
    'Age': [55],           # Numeric: Age in years
    'Sex': [1],            # 1 = Male, 0 = Female (encoded numerically)
    'ChestPainType': [2],  # Encoded values: TA, ATA, NAP, ASY => 0, 1, 2, 3
    'RestingBP': [130],    # Numeric: Resting blood pressure
    'Cholesterol': [250],  # Numeric: Cholesterol level
    'FastingBS': [0],      # Binary: Fasting blood sugar > 120 mg/dL (1 or 0)
    'RestingECG': [1],     # Encoded: Normal, ST, LVH => 0, 1, 2
    'MaxHR': [170],        # Numeric: Maximum heart rate achieved
    'ExerciseAngina': [1], # 1 = Yes, 0 = No (exercise-induced angina)
    'Oldpeak': [1.5],      # Numeric: ST depression induced by exercise
    'ST_Slope': [0]        # Encoded: Up, Flat, Down => 0, 1, 2
})

# Ensure the columns and types are correct
st.write('Input Data Structure and Types:')
st.write(input_data.info())  # To check the data types of each column
st.write("\nInput Data Values:")
st.write(input_data)         # To check the actual values

# Make a prediction using the trained model
try:
    prediction = lr_model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
except Exception as e:
    st.write(f"An error occurred during prediction: {e}")
