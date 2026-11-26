from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# Configure the Streamlit page
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Build paths relative to this app.py file
# Find the folder containing app.py, then look for artifacts beside it
PROJECT_FOLDER = Path(__file__).resolve().parent
ARTIFACTS_FOLDER = PROJECT_FOLDER / "artifacts"

MODEL_PATH = ARTIFACTS_FOLDER / "salary_model.keras"
PREPROCESSING_PATH = ARTIFACTS_FOLDER / "preprocessing.joblib"

# if not MODEL_PATH.is_file() or not PREPROCESSING_PATH.is_file():
#     st.error("One or both model files were not found.")
#     st.write("App folder:", str(PROJECT_FOLDER))
#     st.write("Model path:", str(MODEL_PATH))
#     st.write("Model found:", MODEL_PATH.is_file())
#     st.write("Preprocessing path:", str(PREPROCESSING_PATH))
#     st.write("Preprocessing found:", PREPROCESSING_PATH.is_file())
#     st.stop()


# Load the model and preprocessing objects once, then reuse them
@st.cache_resource
def load_saved_files():
    model = keras.models.load_model(MODEL_PATH)
    artifacts = joblib.load(PREPROCESSING_PATH)
    return model, artifacts


st.title("Customer Salary Prediction")
st.write(
    "Enter the customer details below to get an estimated salary "
    "from the trained regression model."
)

st.info(
    # "This is a classroom project using a sample dataset. "
    "Its prediction is not a real salary assessment."
)

# Stop early with a clear message if Step 9 files are missing
if not MODEL_PATH.exists() or not PREPROCESSING_PATH.exists():
    st.error(
        "Model files are missing. Run Step 9 first and confirm that "
        "the artifacts folder contains both saved files."
    )
    st.stop()

# Load the saved model and preprocessing objects
model, artifacts = load_saved_files()

# Collect customer details
with st.form("salary_prediction_form"):
    st.subheader("Customer details")

    credit_score = st.slider(
    "Credit score",
    min_value=350,
    max_value=850,
    value=650,
    step=1,
    )

    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )

    age = st.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=35,
    step=1,
    )

    tenure = st.slider(
    "Tenure (years)",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
    )

    balance = st.number_input(
        "Account balance",
        min_value=0.0,
        value=50_000.0,
        step=1_000.0
    )

    num_of_products = st.slider(
    "Number of bank products",
    min_value=1,
    max_value=4,
    value=2,
    step=1,
    )

    has_credit_card = st.selectbox(
        "Has a credit card?",
        options=["Yes", "No"]
    )

    is_active_member = st.selectbox(
        "Is an active member?",
        options=["Yes", "No"]
    )

    exited = st.selectbox(
        "Customer exited?",
        options=["No", "Yes"]
    )

    submitted = st.form_submit_button("Predict salary")

# Predict only after the user submits the form
if submitted:
    # Keep names and order consistent with the model's training inputs
    input_values = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_credit_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "Exited": 1 if exited == "Yes" else 0
    }

    input_df = pd.DataFrame([input_values]).reindex(
        columns=artifacts["feature_columns"]
    )

    # Apply the same preprocessing fitted during model training
    processed_input = artifacts["preprocessor"].transform(input_df)
    processed_input = processed_input.astype(np.float32)

    # Predict a scaled salary, then convert it back to salary units
    scaled_prediction = model.predict(processed_input, verbose=0)
    predicted_salary = artifacts["target_scaler"].inverse_transform(
        scaled_prediction
    )[0, 0]

    st.success("Prediction complete")
    st.metric("Estimated salary", f"{predicted_salary:,.2f}")