import os
import json
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import numpy as np
import google.generativeai as genai
from modules.gemini_utils import call_gemini_with_backoff

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "gemini_events.log")

def _append_log_file(entry: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
        'Amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    k = st.slider("Select number of clusters (K)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = model.fit_predict(rfm_scaled)
        
    st.write("Cluster Summary")    
    rfm_summary = rfm.groupby("Cluster").mean()
    rfm_summary['CustomerCount'] = rfm['Cluster'].value_counts().sort_index()
    formatted_df = rfm_summary.style.format("{:.2f}")
    st.dataframe(formatted_df, use_container_width=True)
    
    # Gemini Analysis--------------------------------------------
    rfm_json = rfm_summary.reset_index().to_json(orient='records')

    # initialize session_state items used for concurrency, caching & logging
    st.session_state.setdefault("gemini_in_progress", False)
    st.session_state.setdefault("gemini_api_attempts", 0)
    st.session_state.setdefault("gemini_user_clicks", 0)
    st.session_state.setdefault("gemini_cache", {})
    st.session_state.setdefault("gemini_event_log", [])  # list of dicts

    cache_key = f"gemini_analysis_k{k}"

    # helper to append to session_state event log and to file
    def _log_event(entry: dict):
        st.session_state["gemini_event_log"].append(entry)
        _append_log_file(entry)

    # callback to increment attempts (actual network calls)
    def _increment_attempt():
        st.session_state["gemini_api_attempts"] += 1

    if st.button("🔄 Analyze with Gemini"):
        st.session_state["gemini_user_clicks"] += 1

        # Enforce single request at a time (per session)
        if st.session_state["gemini_in_progress"]:
            st.warning("Another Gemini request is already in progress. Please wait for it to finish.")
        else:
            # If cached, show cached result
            if cache_key in st.session_state["gemini_cache"]:
                st.success("Showing cached Gemini analysis.")
                analysis_text = st.session_state["gemini_cache"][cache_key]
                if analysis_text:
                    st.subheader("🤖 Gemini Analysis of Clusters")
                    st.write(analysis_text)
                else:
                    st.warning("Cached analysis is empty or invalid.")
            else:
                st.session_state["gemini_in_progress"] = True
                try:
                    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_model = genai.GenerativeModel("gemini-2.5-pro")

                    prompt = f"""
                    You are a marketing analyst. The dataset summarizes customer segments derived from a K-Means clustering model based on RFM (Recency, Frequency, Monetary) metrics.
                    Each row represents a customer cluster, showing the average values for Recency, Frequency, and Monetary, along with the total number of customers in that cluster.
               
                    Your tasks are:
                    - Summarize each cluster in 1–2 concise sentences.
                    - Propose a brief, tailored marketing strategy for each cluster.
                    - Use bullet points for clarity.
                    - Present the clusters in order of marketing importance, starting with the most valuable segment.
                            
                    Data:
                    {rfm_json}
                    """
                    # call the helper which logs events and increments attempt counts
                    try:
                        analysis_text = call_gemini_with_backoff(
                            gemini_model,prompt.strip(),max_retries=3,base=2,max_sleep=30,
                            increment_attempt_cb=_increment_attempt,log_event_cb=_log_event,
                        )
                    except Exception as e:
                        _log_event({"timestamp": pd.Timestamp.utcnow().isoformat() + "Z", "event": "final_error", "error": repr(e)})
                        st.error(f"Failed to get analysis from Gemini: {e}")
                        analysis_text = None

                    st.session_state["gemini_cache"][cache_key] = analysis_text

                    if analysis_text:
                        st.subheader("🤖 Gemini Analysis of Clusters")
                        st.write(analysis_text)
                    else:
                        st.warning("No valid analysis returned from Gemini. Check logs below or try again later.")
                finally:
                    st.session_state["gemini_in_progress"] = False

    # show counters and event log in the UI for debugging
    st.write(
        f"User clicks: {st.session_state['gemini_user_clicks']}, "
        f"API attempts (including retries): {st.session_state['gemini_api_attempts']}"
    )

    with st.expander("Show Gemini event log (most recent first)"):
        recent = list(reversed(st.session_state["gemini_event_log"][-50:]))
        for entry in recent:
            ts = entry.get("timestamp", "")
            evt = entry.get("event", "")
            attempt = entry.get("attempt", "")
            msg = f"{ts} | {evt} | attempt={attempt}"
            if "error" in entry:
                msg += f" | error={entry.get('error')}"
            if "extracted_snippet" in entry:
                msg += f" | extracted={entry.get('extracted_snippet')}"
            if "sleep_seconds" in entry:
                msg += f" | sleep={entry.get('sleep_seconds')}"
            st.text(msg)

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

    # ... remaining UI: download button, elbow, silhouette plotting ...
    st.subheader("Download All Cluster Data")
    all_clusters_csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Clusters (CSV)",
        data=all_clusters_csv,
        file_name='all_clusters_data.csv',
        mime='text/csv'
    )

    sse = []
    silhouette_scores = []
    k_range = range(2, 11)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(rfm_scaled)
        sse.append(model.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, labels))

    st.write("### Elbow Method")
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(k_range), y=sse, mode='lines+markers', name='SSE'))
    elbow_fig.update_layout(title='Elbow Method For Optimal K',
                            xaxis_title='Number of clusters (K)',
                            yaxis_title='Sum of Squared Errors (SSE)',
                            width=1400,
                            height=600)
    st.plotly_chart(elbow_fig)

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
