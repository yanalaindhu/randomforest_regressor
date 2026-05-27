import streamlit as st
import numpy as np
import pickle

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Prediction"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = pickle.load(
    open(
        "models/random_forest_model.pkl",
        "rb"
    )
)

# ---------------------------------------------------
# LOAD SCALER
# ---------------------------------------------------
scaler = pickle.load(
    open(
        "models/scaler.pkl",
        "rb"
    )
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("Insurance Charges Prediction")

st.write("Random Forest Regressor Project")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------
sex_value = 1 if sex == "male" else 0

smoker_value = 1 if smoker == "yes" else 0

region_dict = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region_value = region_dict[region]

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("Predict Charges"):

    input_data = np.array([[
        age,
        sex_value,
        bmi,
        children,
        smoker_value,
        region_value
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Insurance Charges = ${prediction[0]:.2f}"
    )