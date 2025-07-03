import streamlit as st
import pandas as pd
import numpy as np
from demand_forecasting import predict_demand
from tsp_solver import tsp_route

st.set_page_config(page_title="Route Optimizer", layout="centered")

st.title("🚚 ML-Based Delivery Route Optimization")

# Section: Choose data input method
option = st.radio("Choose data input method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    st.subheader("🔧 Enter Delivery Data")
    
    num_locations = st.number_input("Number of delivery locations (including depot)", min_value=2, max_value=20, value=5)

    data = []
    for i in range(int(num_locations)):
        with st.expander(f"Location {i+1}"):
            loc = st.text_input(f"Location name {i+1}", f"Loc{i+1}", key=f"loc_{i}")
            time = st.time_input(f"Delivery time {i+1}", key=f"time_{i}")
            load = st.number_input(f"Package load {i+1}", min_value=0, step=1, key=f"load_{i}")
            data.append({"location": loc, "time": time.strftime("%H:%M"), "package_load": load})

    if st.button("Generate Route"):
        df = pd.DataFrame(data)
        st.subheader("Entered Data")
        st.write(df)

        df_pred = predict_demand(df)
        st.subheader("📦 Predicted Demand")
        st.write(df_pred)

        # Distance matrix mockup (replace with haversine or real distances if needed)
        locations = df_pred['location'].tolist()
        n = len(locations)
        np.random.seed(42)
        dist_matrix = np.random.randint(5, 50, size=(n, n))
        np.fill_diagonal(dist_matrix, 0)

        route = tsp_route(locations, dist_matrix)

        st.subheader("🚏 Optimized Route")
        st.success(" → ".join(route))

        st.subheader("📏 Distance Matrix")
        st.dataframe(pd.DataFrame(dist_matrix, index=locations, columns=locations))

else:
    uploaded_file = st.file_uploader("Upload Delivery CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.write(df)

        df_pred = predict_demand(df)
        st.subheader("Predicted Demand")
        st.write(df_pred)

        # Distance matrix
        locations = df_pred['location'].tolist()
        num = len(locations)
        np.random.seed(42)
        dist_matrix = np.random.randint(5, 50, size=(num, num))
        np.fill_diagonal(dist_matrix, 0)

        route = tsp_route(locations, dist_matrix)
        
        st.subheader("Optimized Route")
        st.markdown(" → ".join(route))

        st.subheader("Distance Matrix")
        st.write(pd.DataFrame(dist_matrix, index=locations, columns=locations))
