import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from demand_forecasting import predict_demand
from tsp_solver import tsp_route

# Page setup
st.set_page_config(page_title="Route Optimizer", layout="centered")

# Styling
st.markdown("""
<style>
h1, h2, h3 {
    color: #4B0082;
}
.css-18e3th9 {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚚 ML-Based Delivery Route Optimization")
st.markdown("Welcome to your intelligent delivery assistant powered by ML and TSP 🚀")

# Input mode
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
        st.subheader("📄 Entered Data")
        st.write(df)

        df_pred = predict_demand(df)
        st.subheader("📦 Predicted Demand")
        st.write(df_pred)

        st.subheader("📊 Predicted Demand (Bar Chart)")
        st.bar_chart(df_pred.set_index("location")["predicted_load"])

        locations = df_pred['location'].tolist()
        n = len(locations)
        np.random.seed(42)
        dist_matrix = np.random.randint(5, 50, size=(n, n))
        np.fill_diagonal(dist_matrix, 0)

        with st.spinner("🔄 Optimizing your route..."):
            route = tsp_route(locations, dist_matrix)

        st.success("✅ Route optimization complete!")
        st.balloons()

        st.subheader("🚏 Optimized Route")
        st.success(" → ".join([f"📦 {loc}" for loc in route]))

        st.subheader("📏 Distance Matrix")
        st.dataframe(pd.DataFrame(dist_matrix, index=locations, columns=locations))

        # Add mock coordinates for mapping (use real lat/lon in production)
        coords = pd.DataFrame({
            "location": df_pred['location'],
            "lat": np.linspace(17.40, 17.50, len(df_pred)),
            "lon": np.linspace(78.45, 78.55, len(df_pred))
        })

        st.subheader("🗺️ Route Map")

        route_coords = coords.set_index("location").loc[route].reset_index()

        layer = pdk.Layer(
            "LineLayer",
            data=route_coords,
            get_source_position='[lon, lat]',
            get_target_position='[lon, lat]',
            get_color=[200, 30, 0, 160],
            auto_highlight=True,
            width_scale=3,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=coords['lat'].mean(),
                longitude=coords['lon'].mean(),
                zoom=12,
                pitch=0,
            ),
            layers=[layer]
        ))

else:
    uploaded_file = st.file_uploader("Upload Delivery CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.write(df)

        df_pred = predict_demand(df)
        st.subheader("📦 Predicted Demand")
        st.write(df_pred)

        st.subheader("📊 Predicted Demand (Bar Chart)")
        st.bar_chart(df_pred.set_index("location")["predicted_load"])

        locations = df_pred['location'].tolist()
        num = len(locations)
        np.random.seed(42)
        dist_matrix = np.random.randint(5, 50, size=(num, num))
        np.fill_diagonal(dist_matrix, 0)

        with st.spinner("🔄 Optimizing your route..."):
            route = tsp_route(locations, dist_matrix)

        st.success("✅ Route optimization complete!")
        st.balloons()

        st.subheader("🚏 Optimized Route")
        st.success(" → ".join([f"📦 {loc}" for loc in route]))

        st.subheader("📏 Distance Matrix")
        st.dataframe(pd.DataFrame(dist_matrix, index=locations, columns=locations))

        # Map mock
        coords = pd.DataFrame({
            "location": df_pred['location'],
            "lat": np.linspace(17.40, 17.50, len(df_pred)),
            "lon": np.linspace(78.45, 78.55, len(df_pred))
        })

        st.subheader("🗺️ Route Map")
        route_coords = coords.set_index("location").loc[route].reset_index()

        layer = pdk.Layer(
            "LineLayer",
            data=route_coords,
            get_source_position='[lon, lat]',
            get_target_position='[lon, lat]',
            get_color=[200, 30, 0, 160],
            auto_highlight=True,
            width_scale=3,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=coords['lat'].mean(),
                longitude=coords['lon'].mean(),
                zoom=12,
                pitch=0,
            ),
            layers=[layer]
        ))
