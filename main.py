import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# loading the saved models

diabetes_model = pickle.load(open('D:/multiple Disease Prediction/sav/diabetes_model (2).sav', 'rb'))
diabetes_model_scaler = pickle.load(open('D:/multiple Disease Prediction/sav/diabetes_model_scaler (1).sav', 'rb'))
heart_disease_model = pickle.load(open('D:/multiple Disease Prediction/sav/heart_model.sav', 'rb'))
parkinsons_model = pickle.load(open('D:/multiple Disease Prediction/sav/parkinsons_model.sav', 'rb'))

diabetes_cols = ['Cholesterol', 'Glucose', 'HDL Chol', 'Chol/HDL ratio', 'Age', 'Gender',
                 'Height', 'Weight', 'BMI', 'Systolic BP', 'Diastolic BP', 'waist',
                 'hip', 'Waist/hip ratio']

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    # col1, col2, col3 = st.columns(3)

    Age = st.number_input('Age')
    Gender = st.radio('Gender', ('Male', 'Female'))

    Cholesterol = st.number_input('Cholesterol')
    HDL_Chol = st.number_input('HDL Chol')
    Chol_HDL_ratio = st.number_input('Chol/HDL ratio')

    Glucose = st.number_input('Glucose Level')

    Systolic_BP = st.number_input('Systolic BP')
    Diastolic_BP = st.number_input('Diastolic BP')

    Height = st.number_input('Height')
    Weight = st.number_input('Weight')
    BMI = st.number_input('BMI value')

    waist = st.number_input('Waist')
    hip = st.number_input('Hip')
    Waist_hip_ratio = st.number_input('Waist/hip ratio')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        data = pd.DataFrame([[Cholesterol, Glucose, HDL_Chol, Chol_HDL_ratio, Age, 1 if Gender == 'Male' else 0,
                              Height, Weight, BMI, Systolic_BP, Diastolic_BP, waist,
                              hip, Waist_hip_ratio]], columns=diabetes_cols)

        # standardize the input data
        std_data = diabetes_model_scaler.transform(data)

        diab_prediction = diabetes_model.predict(std_data)

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    # page title
    st.title('Heart Disease Prediction using ML')

    # col1, col2, col3 = st.columns(3)

    age = st.number_input('Age')

    sex = st.radio('Sex', ('Male', 'Female'))

    cp = st.number_input('Chest Pain types')
    trestbps = st.number_input('Resting Blood Pressure')

    chol = st.number_input('Serum Cholestoral in mg/dl')

    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')

    restecg = st.number_input('Resting Electrocardiographic results')

    thalach = st.number_input('Maximum Heart Rate achieved')

    exang = st.number_input('Exercise Induced Angina')

    oldpeak = st.number_input('ST depression induced by exercise')

    slope = st.number_input('Slope of the peak exercise ST segment')

    ca = st.number_input('Major vessels colored by flourosopy')

    thal = st.selectbox('Thal', ('Normal', 'Fixed Defect', 'Reversable Defect'))

    thal = 0 if thal == 'Normal' else 1 if thal == 'Fixed Defect' else 2

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        data = np.asarray(
            [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,
             thal]).reshape(1, -1)
        heart_prediction = heart_disease_model.predict(data)

        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    # col1, col2, col3, col4, col5 = st.columns(5)

    fo = st.number_input('MDVP:Fo(Hz)')

    fhi = st.number_input('MDVP:Fhi(Hz)')

    flo = st.number_input('MDVP:Flo(Hz)')

    Jitter_percent = st.number_input('MDVP:Jitter(%)')

    Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')

    RAP = st.number_input('MDVP:RAP')

    PPQ = st.number_input('MDVP:PPQ')

    DDP = st.number_input('Jitter:DDP')

    Shimmer = st.number_input('MDVP:Shimmer')

    Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')

    APQ3 = st.number_input('Shimmer:APQ3')

    APQ5 = st.number_input('Shimmer:APQ5')

    APQ = st.number_input('MDVP:APQ')

    DDA = st.number_input('Shimmer:DDA')

    NHR = st.number_input('NHR')

    HNR = st.number_input('HNR')

    RPDE = st.number_input('RPDE')

    DFA = st.number_input('DFA')

    spread1 = st.number_input('spread1')

    spread2 = st.number_input('spread2')

    D2 = st.number_input('D2')

    PPE = st.number_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)


