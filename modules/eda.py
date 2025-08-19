import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder

def run(df):
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # Filter out cancelled or adjusted invoices
    df = df[~df['InvoiceNo'].astype(str).str.startswith(('C', 'A'))]

    # Display basic info
    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    stat_df = df[["Quantity", "UnitPrice", "Amount"]].describe().style.format("{:.2f}")
    st.dataframe(stat_df, use_container_width=True)

    # Top 10 selling products (Quantity)
    st.write("### 🏆 Top 10 Selling Products (Quantity)")
    top_products_qty = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_top_products_qty = px.bar(top_products_qty, x='Description', y='Quantity', title="Top 10 Selling Products",width=1400, height=600)
    fig_top_products_qty.update_layout(
        xaxis_tickangle=-90,
        )
    st.plotly_chart(fig_top_products_qty)

    # Top 10 selling products (Amount)
    st.write("### 🏆 Top 10 Selling Products (Amount)")
    top_products_tprice = df.groupby("Description")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_top_products_tprice = px.bar(top_products_tprice, x='Description', y='Amount', title="Top 10 Selling Products",width=1400, height=600)
    fig_top_products_tprice.update_layout(
        xaxis_tickangle=-90,
        )
    st.plotly_chart(fig_top_products_tprice)
 
    # Sales by Country
    st.write("### 🌍 Sales by Country (Top 10)")
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    country_sales = df.groupby("Country")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    fig_country_sales = px.bar(country_sales, x='Country', y='Amount', title="Top 10 Countries by Sales")
    st.plotly_chart(fig_country_sales)

    # Correlation heatmap
    st.write("### 🔍 Correlation heatmap")
    df_encoded = df.copy()
    label_encoders = {}
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            label_encoders[column] = le

    corr_matrix = df_encoded.corr().round(2)
    z = corr_matrix.values
    x = corr_matrix.columns.tolist()
    y = corr_matrix.index.tolist()
    

    #fig_heatmap = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis', showscale=True)

    fig_headmap = ff.create_annotated_heatmap(z,x=x,y=y, colorscale='RdBu', showscale=True, reversescale=True,
                                               zmin=-1,zmax=1,font_colors=['green'])
    #fig_headmap.update_layout(title='Correlation Matrix Heatmap',#autosize=True,width=1200,height=800, 
                              yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_heatmap)

    



