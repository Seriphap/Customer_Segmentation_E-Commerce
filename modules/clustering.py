import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import json
import requests
import google.generativeai as genai

def run(df):
    st.subheader("Customer Segmentation using K-Means")

    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df.dropna(subset=['CustomerID'], inplace=True)

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    k = st.slider("Select number of clusters (K)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm_scaled)
        
    st.write("Cluster Summary")    
    # สร้าง summary ของคลัสเตอร์
    rfm_summary = rfm.groupby("Cluster").mean()
    # เพิ่มจำนวนลูกค้าในแต่ละคลัสเตอร์
    rfm_summary['CustomerCount'] = rfm['Cluster'].value_counts().sort_index()
    # จัดรูปแบบตาราง
    formatted_df = rfm_summary.style.format("{:.2f}")
    # แสดงผลใน Streamlit
    st.dataframe(formatted_df, use_container_width=True)

    # Gemini Analysis--------------------------------------------
    rfm_json = rfm_summary.reset_index().to_json(orient='records')

    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # Initialize Gemini client
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Create prompt

    prompt = f"""
    You are a marketing analyst. Below is a customer segmentation summary from a K-Means clustering model using RFM (Recency, Frequency, Monetary) data.
    
    Each row represents a cluster of customers with average values for Recency, Frequency, and Monetary, along with the number of customers in that cluster.
    
    Please:
    1. Describe the characteristics of each cluster.
    2. Suggest a marketing strategy for each cluster.
    
    Data:
    {rfm_json}
    """

    
    # Send request to Gemini
    response = model.generate_content(prompt.strip())
    
    # Display result
    st.subheader("Gemini Analysis of Clusters")
    st.write(response.text)

    #------------------------------------------------------------
    
    st.session_state['rfm'] = rfm
    st.session_state['rfm_scaled'] = rfm_scaled
    st.session_state['model'] = model

    st.subheader("RFM Variable Descriptions")
    text_explain = {
            "ตัวแปร (Variable)": ["Recency", "Frequency", "Monetary"],
            "คำอธิบายภาษาไทย (Thai Description)": [
                "จำนวนวันนับจากการซื้อครั้งล่าสุด (น้อย = ซื้อเร็ว ๆ นี้)",
                "จำนวนครั้งที่ซื้อ (มาก = ซื้อบ่อย)",
                "ยอดใช้จ่ายรวม (มาก = ลูกค้ามูลค่าสูง)"
            ],
            "English Description": [
                "Number of days since the last purchase (lower = more recent)",
                "Number of purchases (higher = more frequent)",
                "Total spending amount (higher = more valuable customer)"
            ]
        }
    st.dataframe(pd.DataFrame(text_explain),hide_index=True)

    # ... download buttons for each cluster ...
    st.subheader("Download All Cluster Data")

    # Reset index to include CustomerID in the CSV
    all_clusters_csv = rfm.reset_index().to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download All Clusters (CSV)",
        data=all_clusters_csv,
        file_name='all_clusters_data.csv',
        mime='text/csv'
    )

    # Elbow method & Silhouette score
    sse = []
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(rfm_scaled)
        sse.append(model.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, labels))

    # Plot Elbow Method using Plotly
    st.write("### Elbow Method")
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(k_range), y=sse, mode='lines+markers', name='SSE'))
    elbow_fig.update_layout(title='Elbow Method For Optimal K',
                            xaxis_title='Number of clusters (K)',
                            yaxis_title='Sum of Squared Errors (SSE)',
                            width=1400,
                            height=600)
    st.plotly_chart(elbow_fig)

    # Plot Silhouette Score using Plotly
    st.write("### Silhouette Score")
    silhouette_fig = go.Figure()
    silhouette_fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                                        name='Silhouette Score', line=dict(color='green')))
    silhouette_fig.update_layout(title='Silhouette Score For Optimal K',
                                 xaxis_title='Number of clusters (K)',
                                 yaxis_title='Silhouette Score',
                                width=1400,
                                height=600)
    st.plotly_chart(silhouette_fig)




 
