# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:00:54 2024

@author: Navaneeth G
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set page configuration
st.set_page_config(page_title="Insurance Prediction App",
                   layout="wide",
                   page_icon="💰")

# Load the saved model
model_path = "insurance_model.sav"
insurance_model = pickle.load(open("C:/Users/Navaneeth G/OneDrive/Desktop/newproject/insurance_model.sav", 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Insurance Prediction System',
        ['Insurance Prediction'],
        menu_icon='bar-chart-line',
        icons=['currency-dollar'],
        default_index=0
    )

# Insurance Prediction Page
if selected == 'Insurance Prediction':
    # Page title
    st.title("Insurance Charges Prediction")

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age_input = st.text_input('Age (e.g., 25)')

    with col2:
        bmi_input = st.text_input('BMI (e.g., 24.5)')

    with col3:
        children_input = st.text_input('Number of Children (e.g., 2)')

    with col1:
        sex_male_input = st.selectbox('Sex', ['Female', 'Male'])
        sex_male_input = 1 if sex_male_input == 'Male' else 0

    with col2:
        smoker_yes_input = st.selectbox('Smoker', ['No', 'Yes'])
        smoker_yes_input = 1 if smoker_yes_input == 'Yes' else 0

    # Prediction Code
    prediction_result = ""

    if st.button("Predict Insurance Charges"):
        # Validate input data
        try:
            age_input = float(age_input)
            bmi_input = float(bmi_input)
            children_input = int(children_input)

            # Create input array
            user_input = np.array([age_input, bmi_input, children_input, sex_male_input, smoker_yes_input]).reshape(1, -1)

            # Predict using the model
            Future_Prediction = insurance_model.predict(user_input)
            prediction_result = f"Predicted Insurance Charges: ${Future_Prediction[0]:,.2f}"

        except ValueError:
            prediction_result = "Please enter valid numeric values for Age, BMI, and Children."

    st.success(prediction_result)
