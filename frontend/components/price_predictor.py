import pickle
import numpy as np
import pandas as pd
import streamlit as st
import requests


st.set_page_config(page_title="Price Predictor", page_icon="📈", layout="wide")



# HuggingFace URLs for your files
MODEL_URL = "https://huggingface.co/sayan2006/realestate-models/resolve/main/pipeline.pkl"
DATA_URL  = "https://huggingface.co/sayan2006/realestate-models/resolve/main/df.pkl"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    response = requests.get(MODEL_URL)
    return pickle.loads(response.content)   # load directly from bytes

@st.cache_data(show_spinner="Loading data...")
def load_df():
    response = requests.get(DATA_URL)
    return pickle.loads(response.content)   # load directly from bytes

# Use them
model = load_model()
df = load_df()

# ---------------------------------------------------------------- header
st.title("📈 Property Price Predictor")
st.caption("Fill in the property details and get an estimated price range for a Kolkata property.")

# ---------------------------------------------------------------- inputs
with st.form("predict_form"):

    # --- Section 1: size & rooms
    with st.container(border=True):
        st.markdown("#### 🛏️ Size & rooms")
        a1, a2, a3, a4 = st.columns(4)
        bedroom_num = a1.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        balcony_num = a2.number_input("Balconies", min_value=0, max_value=10, value=1)
        area_sqft = a3.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)
        age = a4.number_input("Age (years)", min_value=0, max_value=100, value=5)

    # --- Section 2: building details
    with st.container(border=True):
        st.markdown("#### 🏢 Building details")
        b1, b2, b3, b4 = st.columns(4)
        total_floor = b1.number_input("Total Floors", min_value=0, max_value=100, value=10)
        floor_num = b2.number_input("Floor Number", min_value=0, max_value=100, value=2)
        total_landmark_count = b3.number_input("Total Landmark Count", min_value=0, max_value=50, value=5)
        cooperative_society = b4.selectbox("Cooperative Society", ["Yes", "No"])
        cooperative_society = 1 if cooperative_society == "Yes" else 0

    # --- Section 3: features
    with st.container(border=True):
        st.markdown("#### 🛋️ Features")
        c1, c2, c3 = st.columns(3)
        facing_label = c1.selectbox("Kolkata Area", df["FACING_LABEL"].unique())
        furnish_label = c2.selectbox("Furnishing", df["FURNISH_LABEL"].unique())
        property_type = c3.selectbox("Property Type", df["PROPERTY_type"].unique())

    # --- Section 4: location
    with st.container(border=True):
        st.markdown("#### 📍 Location")
        d1, d2, d3 = st.columns(3)
        city_labels = d1.selectbox("City", df["CITY_LABELS"].unique())
        city_num = d2.selectbox("City Number", sorted(df["CITY_NUM"].unique().tolist()))
        locality = d3.selectbox("Locality", df["LOCALITY"].unique())

        e1, e2 = st.columns(2)
        latitude = e1.number_input("Latitude", value=22.572600, format="%.6f")
        longitude = e2.number_input("Longitude", value=88.363900, format="%.6f")

    submitted = st.form_submit_button("Predict price", type="primary", width = "stretch")

# ---------------------------------------------------------------- prediction
if submitted:

    data = [[bedroom_num, age, total_floor, total_landmark_count, balcony_num, floor_num,
             area_sqft, latitude, longitude, cooperative_society, facing_label, furnish_label,
             property_type, city_labels, city_num, locality]]
    columns = ['BEDROOM_NUM', 'AGE', 'TOTAL_FLOOR', 'TOTAL_LANDMARK_COUNT',
               'BALCONY_NUM', 'FLOOR_NUM', 'AREA_SQFT', 'LATITUDE', 'LONGITUDE',
               'Cooperative Society', 'FACING_LABEL', 'FURNISH_LABEL', 'PROPERTY_type',
               'CITY_LABELS', 'CITY_NUM', 'LOCALITY']

    one_df = pd.DataFrame(data, columns=columns)

    base_price = np.expm1(model.predict(one_df))[0]

    margin = 0.10
    lower = base_price * (1 - margin)
    upper = base_price * (1 + margin)

    with st.container(border=True):
        st.markdown("#### 💰 Estimated price")
        r1, r2, r3 = st.columns(3)
        r1.metric("Lower estimate", f"₹{lower:.2f} Cr")
        r2.metric("Expected price", f"₹{base_price:.2f} Cr")
        r3.metric("Upper estimate", f"₹{upper:.2f} Cr")

        st.success(f"The price of the property is between ₹{lower:.2f} Cr and ₹{upper:.2f} Cr")
        st.caption(f"Range is the model estimate ±{margin * 100:.0f}%.")