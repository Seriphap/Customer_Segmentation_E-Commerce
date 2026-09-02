import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import LabelEncoder


def run(df):

    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # ==================================================
    # Data Cleaning
    # ==================================================

    # Remove cancelled / adjusted invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith(("C", "A"))].copy()

    # Create Amount column BEFORE using it
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    # ==================================================
    # Sample Data
    # ==================================================

    st.write("### Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    # ==================================================
    # Summary Statistics
    # ==================================================

    st.write("### Summary Statistics")

    stat_df = (
        df[["Quantity", "UnitPrice", "Amount"]]
        .describe()
        .round(2)
    )

    st.dataframe(
        stat_df,
        use_container_width=True
    )

    # ==================================================
    # Top Products by Quantity
    # ==================================================

    st.write("### 🏆 Top 10 Selling Products (Quantity)")

    top_products_qty = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_top_products_qty = px.bar(
        top_products_qty,
        x="Description",
        y="Quantity",
        title="Top 10 Selling Products (Quantity)",
        width=1400,
        height=600
    )

    fig_top_products_qty.update_layout(
        xaxis_tickangle=-90
    )

    st.plotly_chart(
        fig_top_products_qty,
        use_container_width=True
    )

    # ==================================================
    # Top Products by Amount
    # ==================================================

    st.write("### 🏆 Top 10 Selling Products (Amount)")

    top_products_amount = (
        df.groupby("Description")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_top_products_amount = px.bar(
        top_products_amount,
        x="Description",
        y="Amount",
        title="Top 10 Selling Products (Amount)",
        width=1400,
        height=600
    )

    fig_top_products_amount.update_layout(
        xaxis_tickangle=-90
    )

    st.plotly_chart(
        fig_top_products_amount,
        use_container_width=True
    )

    # ==================================================
    # Sales by Country
    # ==================================================

    st.write("### 🌍 Sales by Country (Top 10)")

    country_sales = (
        df.groupby("Country")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_country_sales = px.bar(
        country_sales,
        x="Country",
        y="Amount",
        title="Top 10 Countries by Sales"
    )

    st.plotly_chart(
        fig_country_sales,
        use_container_width=True
    )

    # ==================================================
    # Correlation Heatmap
    # ==================================================

    st.write("### 🔍 Correlation Heatmap")

    df_encoded = df.copy()

    for column in df_encoded.columns:

        if (
            pd.api.types.is_object_dtype(df_encoded[column])
            or pd.api.types.is_string_dtype(df_encoded[column])
            or pd.api.types.is_categorical_dtype(df_encoded[column])
        ):

            le = LabelEncoder()

            df_encoded[column] = le.fit_transform(
                df_encoded[column]
                .fillna("Unknown")
                .astype(str)
            )

    # Select numeric columns only
    numeric_df = df_encoded.select_dtypes(
        include=["number", "bool"]
    )

    if numeric_df.shape[1] < 2:

        st.warning(
            "Not enough numerical columns to generate a correlation matrix."
        )

    else:

        corr_matrix = (
            numeric_df
            .corr()
            .round(2)
        )

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Matrix"
        )

        fig_heatmap.update_layout(
            height=900
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

    












