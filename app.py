
# IMPORT LIBRARIES

import streamlit as st
import joblib
import pandas as pd

#python -m streamlit run app.py
# LOAD MODEL AND ENCODERS

model = joblib.load(
    "best_model.pkl"
)

le_fert = joblib.load(
    "encoder.pkl"
)

le_soil = joblib.load(
    "encoder_soil.pkl"
)

le_crop = joblib.load(
    "encoder_crop.pkl"
)



# PAGE CONFIGURATION

st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)



# TITLE
st.title(
    "Fertilizer Recommendation System"
)

st.subheader(
    "Smart Agriculture for Ethiopian Farmers"
)

st.write(
    "Enter soil and environmental information to predict the best fertilizer."
)


# INPUT SECTION

# ROW 1
col1, col2, col3 = st.columns(3)

with col1:

    soil_type = st.selectbox(
        "Soil Type",
        le_soil.classes_
    )

with col2:

    soil_pH = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

with col3:

    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        value=40.0
    )


# ROW 2
col4, col5, col6 = st.columns(3)

with col4:

    nitrogen = st.number_input(
        "Nitrogen Level",
        min_value=0.0,
        value=35.0
    )

with col5:

    phosphorus = st.number_input(
        "Phosphorus Level",
        min_value=0.0,
        value=20.0
    )

with col6:

    potassium = st.number_input(
        "Potassium Level",
        min_value=0.0,
        value=15.0
    )


# ROW 3
col7, col8, col9 = st.columns(3)

with col7:

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        value=28.0
    )

with col8:

    humidity = st.number_input(
        "Humidity",
        min_value=0.0,
        value=65.0
    )

with col9:

    rainfall = st.number_input(
        "Rainfall",
        min_value=0.0,
        value=120.0
    )


# ROW 4
crop_type = st.selectbox(
    "Crop Type",
    le_crop.classes_
)


# PREDICTION BUTTON

if st.button("🚀 Predict Fertilizer"):

    try:

        # Encode categorical features
        soil_encoded = le_soil.transform(
            [soil_type]
        )[0]

        crop_encoded = le_crop.transform(
            [crop_type]
        )[0]

        # Create dataframe
        input_data = pd.DataFrame([{

            'Soil_Type': soil_encoded,

            'Soil_pH': soil_pH,

            'Soil_Moisture': soil_moisture,

            'Nitrogen_Level': nitrogen,

            'Phosphorus_Level': phosphorus,

            'Potassium_Level': potassium,

            'Temperature': temperature,

            'Humidity': humidity,

            'Rainfall': rainfall,

            'Crop_Type': crop_encoded

        }])

        # Predict
        prediction = model.predict(
            input_data
        )

        # Decode prediction
        result = le_fert.inverse_transform(
            prediction
        )[0]

        # Display result
        st.success(
            f" Recommended Fertilizer: {result}"
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


