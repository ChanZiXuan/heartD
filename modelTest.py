import streamlit as st
import pandas as pd
from joblib import load
import joblib

# Load the logistic regression model with preprocessing steps included
lr_model = load('logisticregression2.joblib')


def main():
    st.title("Heart Disease Prediction")

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

    st.write("Input Data:", input_data)

    # Check for NaN values in the input data
    if input_data.isnull().values.any():
        st.write("Error: Missing values detected in input data.")
        return

    if st.button("Predict Heart Disease"):
        try:
            # Preprocessing and Prediction
            # Apply the model pipeline, which includes any preprocessing (e.g., scaling, encoding)
            prediction = lr_model.predict(input_data)

            # Display the prediction result
            if prediction[0] == 1:
                st.write('The model predicts that this person has heart disease.')
            else:
                st.write('The model predicts that this person does not have heart disease.')
        except Exception as e:
            st.write(f"An error occurred during prediction: {e}")

# Ensure the main function is called
if __name__ == "__main__":
    main()
