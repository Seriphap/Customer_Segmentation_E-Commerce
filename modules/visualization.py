import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

def run(df):
    st.subheader("3D Cluster Visualization")

    if 'rfm_scaled' not in st.session_state or 'rfm' not in st.session_state or 'model' not in st.session_state:
        st.warning("Please run clustering first.")
        return

    rfm_scaled = st.session_state['rfm_scaled']
    rfm = st.session_state['rfm'].copy()
    k = st.session_state['model'].n_clusters

    method = st.selectbox("Select 3D visualization method", ["3D RFM", "3D PCA", "3D t-SNE"])

    if method == "3D RFM":
        if all(col in rfm.columns for col in ['Recency', 'Frequency', 'Monetary']):
            fig = px.scatter_3d(
                rfm,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color=rfm['Cluster'].astype(str),
                title=f"3D RFM Clustering (K={k})",
                labels={
                    'Recency': 'Recency (days)',
                    'Frequency': 'Frequency (purchases)',
                    'Monetary': 'Monetary (spending)'
                }
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(width=1400, height=600)
            st.plotly_chart(fig)
        else:
            st.error("RFM columns not found in data.")

    elif method == "3D PCA":
        pca = PCA(n_components=3)
        components = pca.fit_transform(rfm_scaled)
        rfm['PCA1'], rfm['PCA2'], rfm['PCA3'] = components[:, 0], components[:, 1], components[:, 2]
        fig = px.scatter_3d(
            rfm,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color=rfm['Cluster'].astype(str),
            title=f"3D PCA Clustering (K={k})"
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(width=1400, height=600)
        st.plotly_chart(fig)

    elif method == "3D t-SNE":
        tsne = TSNE(n_components=3, random_state=42, perplexity=30,  max_iter=1000)
        tsne_components = tsne.fit_transform(rfm_scaled)
        rfm['TSNE1'], rfm['TSNE2'], rfm['TSNE3'] = tsne_components[:, 0], tsne_components[:, 1], tsne_components[:, 2]
        fig = px.scatter_3d(
            rfm,
            x='TSNE1',
            y='TSNE2',
            z='TSNE3',
            color=rfm['Cluster'].astype(str),
            title=f"3D t-SNE Clustering (K={k})"
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(width=1400, height=600)
        st.plotly_chart(fig)
