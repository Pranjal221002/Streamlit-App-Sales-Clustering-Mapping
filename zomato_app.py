import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from io import BytesIO

# Function to preprocess data
def preprocess_data(file):
    chunk_size = 1000
    chunks = pd.read_csv(file, encoding='ISO-8859-1', on_bad_lines='skip', chunksize=chunk_size)
    df = pd.concat(chunks, ignore_index=True)
    required_columns = ['Restaurant ID', 'Latitude', 'Longitude', 'Sales (in USD)', 'Aggregate rating']
    df = df[required_columns].dropna()
    return df

# Function to perform K-Means clustering
def perform_clustering(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    df['Cluster'] = df['Cluster'].astype(int) 
    return df

# Function to create Folium map
def create_map(df):
    cluster_colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen"]
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
    
    for _, row in df.iterrows():
        # Cast the 'Cluster' value to an integer before using it as an index
        cluster_index = int(row['Cluster']) % len(cluster_colors)
        
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            color=cluster_colors[cluster_index],
            fill=True,
            fill_color=cluster_colors[cluster_index],
            fill_opacity=0.7,
            popup=f"Restaurant ID: {row['Restaurant ID']}\nSales: {row['Sales (in USD)']}\nRating: {row['Aggregate rating']}"
        ).add_to(m)
    
    return m


def create_excel_file(df):
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name="Clustered_Data")
    output.seek(0)
    return output


def main():
    st.title("K-Means Clustering on Sales Data")
    st.sidebar.header("Upload Your Sales Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    num_clusters = st.sidebar.slider("Select Number of Clusters", 1, 10, 3)
    
    if uploaded_file:
        df = preprocess_data(uploaded_file)
        df = perform_clustering(df, num_clusters)
        
        st.subheader("Clustered Data")
        st.dataframe(df)
        
        st.subheader("Clustered Map")
        map_object = create_map(df)
        folium_static(map_object)
        
        st.subheader("Download Clustered Data")
        excel_file = create_excel_file(df)
        st.download_button(label="Download Excel", data=excel_file, file_name="Clustered_Sales_Data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
