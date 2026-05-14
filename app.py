import sys
import streamlit as st
import joblib
# Load the model and scaler
#model = joblib.load('heart_failure_model.pkl')

st.title('Heart Failure Prediction')

age = st.slider('Age', 40, 95, 60)
anaemia = st.selectbox('Anaemia', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase', min_value=0, max_value=10000, value=500)
diabetes = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
ejection_fraction = st.slider('Ejection Fraction', 0, 100, 30)
high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
platelets = st.number_input('Platelets', min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0,max_value=2.0, value=1.0, step=0.1)
serum_sodium = st.number_input('Serum Sodium', min_value=0, max_value=200, value=135)
sex = st.radio('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
smoking = st.selectbox('Smoking', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')   
time = st.number_input('Follow-up Time', min_value=0, max_value=500, value=100)

input_data = [[age, anaemia, creatinine_phosphokinase, diabetes, 
               ejection_fraction, high_blood_pressure, platelets, serum_creatinine, 
               serum_sodium, sex, smoking, time]]

if st.button('Predict'):
    # Scale the input data
    scaler = joblib.load('scaler.pkl')
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    model = joblib.load('heart_failure_model.pkl')
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.info('The patient is likely to experience heart failure.')
    else:
        st.info('The patient is unlikely to experience heart failure.')