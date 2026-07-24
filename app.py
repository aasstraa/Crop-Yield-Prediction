import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Prediction Model")
st.markdown("Predict crop yield based on environmental & farming factors")
st.markdown("---")

# ===== LOAD MODEL PIPELINE =====
@st.cache_resource
def load_model():
    # This loaded file contains your complete pipeline: OneHotEncoder + LinearRegression
    return joblib.load('crop_yield_model.pkl')

model = load_model()

# ===== USER INTERFACE OPTIONS =====
# These arrays populate your drop-down menus
region_options = ['West', 'South', 'North', 'East']
soil_options = ['Sandy', 'Clay', 'Loam', 'Silt', 'Peaty', 'Chalky']
crop_options = ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize']
weather_options = ['Cloudy', 'Rainy', 'Sunny']

# ===== INPUT SECTION =====
st.subheader("📊 Enter Crop Parameters")

col1, col2 = st.columns(2)

# Column 1: Object/Categorical drop-down menus
with col1:
    region = st.selectbox("📍 Region", region_options)
    soil_type = st.selectbox("🪨 Soil Type", soil_options)
    crop = st.selectbox("🌾 Crop Type", crop_options)
    weather = st.selectbox("🌤️ Weather Condition", weather_options)

# Column 2: Continuous/Numeric features
with col2:
    Rainfall_mm = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=5.0
    )
    Temperature_Celsius = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=0.5
    )
    Days_to_Harvest = st.number_input(
        "📅 Days to Harvest",
        min_value=30,
        max_value=365,
        value=120
    )

# Binary/Categorical Radio Buttons (mapped to continuous 0/1 integers)
col3, col4 = st.columns(2)
with col3:
    fertilizer = st.radio("🧪 Fertilizer Used?", ["No", "Yes"], index=0)
    Fertilizer_Used = 1 if fertilizer == "Yes" else 0

with col4:
    irrigation = st.radio("💧 Irrigation Used?", ["No", "Yes"], index=0)
    Irrigation_Used = 1 if irrigation == "Yes" else 0

st.markdown("---")

# ===== PREPARE FEATURES =====

input_data = pd.DataFrame([{
    'Region': region,
    'Soil_Type': soil_type,
    'Crop': crop,
    'Rainfall_mm': Rainfall_mm,
    'Temperature_Celsius': Temperature_Celsius,
    'Fertilizer_Used': Fertilizer_Used,
    'Irrigation_Used': Irrigation_Used,
    'Weather_Condition': weather,
    'Days_to_Harvest': Days_to_Harvest
}])

# ===== PREDICTION =====
if st.button("🔮 Predict Yield", use_container_width=True, type="primary"):
    try:
        # Pass the DataFrame to the pipeline. 
        # The OneHotEncoder inside the pipeline will process the string columns seamlessly.
        prediction = model.predict(input_data)[0]
        
        # Display completion state
        st.success("✅ Prediction Complete!")
        
        col_result1, col_result2 = st.columns(2)
        with col_result1:
            st.metric("Predicted Yield", f"{prediction:.2f} tons/ha")
        
        with col_result2:
            st.metric("Crop", crop)
        
        # Expandable input configuration summary
        with st.expander("📋 Input Summary"):
            st.write(f"""
            - **Region:** {region}
            - **Soil Type:** {soil_type}
            - **Crop:** {crop}
            - **Rainfall:** {Rainfall_mm} mm
            - **Temperature:** {Temperature_Celsius}°C
            - **Fertilizer Used:** {'Yes' if Fertilizer_Used else 'No'}
            - **Irrigation Used:** {'Yes' if Irrigation_Used else 'No'}
            - **Weather Condition:** {weather}
            - **Days to Harvest:** {Days_to_Harvest}
            """)
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.warning(
            "⚠️ Mismatch Detected: Verify that the keys in the `input_data` DataFrame "
            "above exactly match your model's training dataframe (`x_train`) feature names and order."
        )
