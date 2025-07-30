import streamlit as st
import pandas as pd
import gdown
from modules import eda, clustering, visualization

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🛍️ Customer Segmentation for E-Commerce")

@st.cache_data
def load_data():
    
    file_id = '1kin915XqIlc5pSuyDjp6nuc-QXkDx-d6'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'data.csv'
    
    # Download the file if not already present
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    # Load and prepare the data
    df = pd.read_csv(output)
    df = df[~df['InvoiceNo'].astype(str).str.startswith(('C', 'A'))]
    df = df[df['Quantity'].astype(int) >= 0]
    df.columns = df.columns.str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

menu = st.sidebar.radio("Select Section", ["EDA", "Clustering", "Visualization"])

if menu == "EDA":
    eda.run(df)
elif menu == "Clustering":
    clustering.run(df)
elif menu == "Visualization":
    visualization.run(df)

