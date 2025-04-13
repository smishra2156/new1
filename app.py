import streamlit as st
import numpy as np
import pickle

# Title
st.title("Diabetes Prediction App")
st.write("Enter health information to predict the likelihood of diabetes.")
# Download the pipeline if not already downloaded
#file_path = "diabetes_pipeline.pkl"
#if not os.path.exists(file_path):
 #   st.info("Downloading model file...")
#    url = "https://drive.google.com/uc?id=1aE_daexQbhtTrtFBThxK1_ytCvGaKn_i"  # due to size issue uploaded file on my google drive
#    gdown.download(url, file_path, quiet=False)
# Load the pipeline
try:
    #pipeline = joblib.load(file_path) scaler
    #pipeline = joblib.load("scaler.pkl")
    with open("scaler.pkl", 'rb') as file:
        pipeline = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# User input form
with st.form("health_form"):
    st.subheader("Health & Lifestyle Inputs")

    HighBP = st.selectbox("Do you have high blood pressure?", ["No", "Yes"])
    HighChol = st.selectbox("Do you have high cholesterol?", ["No", "Yes"])
    CholCheck = st.selectbox("Have you had a cholesterol check in the past 5 years?", ["No", "Yes"])
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0)
    Smoker = st.selectbox("Have you smoked at least 100 cigarettes in your life?", ["No", "Yes"])
    Stroke = st.selectbox("Have you ever had a stroke?", ["No", "Yes"])
    HeartDiseaseorAttack = st.selectbox("Do you have coronary heart disease or myocardial infarction?", ["No", "Yes"])
    PhysActivity = st.selectbox("Have you exercised in the past 30 days?", ["No", "Yes"])
    HvyAlcoholConsump = st.selectbox("Do you consume more than 14 drinks/week (men) or 7/week (women)?", ["No", "Yes"])
    AnyHealthcare = st.selectbox("Do you have any form of healthcare coverage?", ["No", "Yes"])
    NoDocbcCost = st.selectbox("Was there a time you could not see a doctor due to cost?", ["No", "Yes"])
    GenHlth = st.slider("How would you rate your general health?", min_value=1, max_value=5, value=3, help="1=Excellent, 5=Poor")
    MentHlth = st.slider("Number of days your mental health was not good (last 30 days)", 0, 30, 0)
    PhysHlth = st.slider("Number of days your physical health was not good (last 30 days)", 0, 30, 0)
    DiffWalk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes"])
    Sex = st.selectbox("What is your sex?", ["Male", "Female"])
    Age = st.slider("Age Category", min_value=1, max_value=13, value=7, help="1=18-24, 13=80+")

    submitted = st.form_submit_button("Submit")

if submitted:
    # Convert Yes/No to 1/0
    def yn_to_binary(x): return 1 if x == "Yes" else 0
    def sex_to_binary(x): return 1 if x == "Male" else 0

    input_data = np.array([[
        yn_to_binary(HighBP),
        yn_to_binary(HighChol),
        yn_to_binary(CholCheck),
        BMI,
        yn_to_binary(Smoker),
        yn_to_binary(Stroke),
        yn_to_binary(HeartDiseaseorAttack),
        yn_to_binary(PhysActivity),
        yn_to_binary(HvyAlcoholConsump),
        yn_to_binary(AnyHealthcare),
        yn_to_binary(NoDocbcCost),
        GenHlth,
        MentHlth,
        PhysHlth,
        yn_to_binary(DiffWalk),
        sex_to_binary(Sex),
        Age,
    ]])

    st.write("### Input Array for Model:")
    st.write(input_data)

try:
    input_data_as_np_array = np.array(input_data)[0]
    input_data_as_np_reshaped = input_data_as_np_array.reshape(1, -1)
    input_data_scaled = pipeline.transform(input_data_as_np_reshaped)
    prediction = pipeline.predict(input_data_scaled)[0]
    #prediction = pipeline.predict(input_data)[0]
    #prob = pipeline.predict_proba(input_data)[0][1]
    st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    #st.write(f"Probability of Diabetes: {prob:.2%}")
except Exception as e:
    st.error(f"Prediction failed: {e}")