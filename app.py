import streamlit as st
import pandas as pd
import numpy as np
from demand_forecasting import predict_demand
from tsp_solver import tsp_route

st.title("🚚 ML-Based Delivery Route Optimization")

uploaded_file = st.file_uploader("Upload Delivery CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df)

    df_pred = predict_demand(df)
    st.subheader("Predicted Demand")
    st.write(df_pred)

    # Create mock distance matrix
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
