import pandas as pd
from joblib import load
import joblib 

# Load your trained model
lr_model = load('logisticregression2.joblib')

def main():
    st.title('Heart Disease Prediction')
# Define a sample input similar to what you would get from the Streamlit form
input_data = pd.DataFrame({
    'Age': [55],           # Numeric
    'Sex': [1],            # 1 = Male, 0 = Female (ensure this is numeric)
    'ChestPainType': [2],  # Encoded as 0, 1, 2, 3 (make sure you use the correct numeric encoding)
    'RestingBP': [130],    # Numeric
    'Cholesterol': [250],  # Numeric
    'FastingBS': [0],      # Binary 0 or 1 (numeric)
    'RestingECG': [1],     # Encoded as 0, 1, 2
    'MaxHR': [170],        # Numeric
    'ExerciseAngina': [1], # 1 = Yes, 0 = No (numeric)
    'Oldpeak': [1.5],      # Numeric with decimal
    'ST_Slope': [0]        # Encoded as 0, 1, 2
})

# Check the input data
print("Input Data:")
print(input_data)

# Attempt prediction using the trained model
try:
    prediction = lr_model.predict(input_data)
    # Show the result
    if prediction[0] == 1:
        st.write('The model predicts that this person has heart disease.')
    else:
        st.write('The model predicts that this person does not have heart disease.')
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"An error occurred during prediction: {e}")
