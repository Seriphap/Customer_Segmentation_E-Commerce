import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import plotly.graph_objects as go
import requests
import time
import json

def run(df: pd.DataFrame):
    st.subheader("Customer Segmentation using K-Means")

    # ========= 0) Validate & Preprocess =========
    required_cols = {"Quantity", "UnitPrice", "InvoiceNo", "CustomerID", "InvoiceDate", "Amount"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        return

    # Ensure datetime
    if not np.issubdtype(df["InvoiceDate"].dtype, np.datetime64):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Basic cleaning
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])

    # RFM base (ยังไม่รวม Cluster)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
    )

    # ========= 1) UI: เลือก K และปุ่มรัน (กัน rerun) =========
    with st.form("rfm_form"):
        k = st.slider("Select number of clusters (K)", 2, 10, 4)
        submitted = st.form_submit_button("Run clustering & explain")

    # ========= 2) Helper: backoff + prompt + cache =========
    def call_gemini_with_backoff(model, prompt: str,
                                 max_retries: int = 5,
                                 base: int = 2,
                                 max_sleep: int = 30):
        """Retry เมื่อเจอ 429 RESOURCE_EXHAUSTED ด้วย truncated exponential backoff"""
        for i in range(max_retries):
            try:
                return model.generate_content(prompt)
            except ResourceExhausted:
                sleep = min(max_sleep, base ** i)
                time.sleep(sleep)
        # ถ้ายังล้มเหลว ให้โยนต่อ
        raise

    def build_explain_prompt(summary_rows):
        """
        summary_rows: list[dict] เช่น
        [{"Cluster":0,"Recency":43.70,"Frequency":3.68,"Monetary":1359.05,"CustomerCount":3054}, ...]
        """
        return f"""
        คุณเป็นผู้ช่วยด้านการตลาด ช่วยอธิบายผลการจัดกลุ่มลูกค้าจาก RFM (Recency, Frequency, Monetary)
        โดยอิงจากสรุปด้านล่าง (ค่าเฉลี่ยต่อคลัสเตอร์) และเสนอแผนการตลาดแบบสั้น กระชับ
        
        ข้อกำหนด:
        - ตั้งชื่อกลุ่มแบบเข้าใจง่าย (เช่น 'ลูกค้าค่าคงที่สูงและมาซื้อบ่อย', 'เสี่ยงหลุด' ฯลฯ)
        - อธิบายแต่ละคลัสเตอร์ 1-2 ประโยค
        - แนะนำแคมเปญ 2 ข้อต่อคลัสเตอร์ (ช่องทาง + ข้อเสนอ) เป็น bullet point
        - ภาษา: ไทย
        - หมายเหตุ RFM: Recency ยิ่งน้อย = มาซื้อล่าสุด; Frequency ยิ่งมาก = มาซื้อบ่อย; Monetary ยิ่งมาก = มูลค่าซื้อสูง
        
        Cluster summary (JSON):
        {json.dumps(summary_rows, ensure_ascii=False)}
        """.strip()

    @st.cache_data(show_spinner=False, ttl=3600)
    def explain_cached(model_name: str, summary_key: tuple, summary_rows: list):
        """แคชผลลัพธ์ตาม summary (ป้องกันยิงซ้ำเมื่อข้อมูลเดิม)"""
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_name)
        prompt = build_explain_prompt(summary_rows)
        resp = call_gemini_with_backoff(model, prompt)
        return resp.text

    # ========= 3) เมื่อกด Submit: ทำ KMeans + Summary + เรียก AI =========
    if submitted:
        # Scale & Fit
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)

        # ตั้ง n_init ให้เหมาะกับ scikit-learn ใหม่ ๆ
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(rfm_scaled)

        rfm_with_cluster = rfm.copy()
        rfm_with_cluster["Cluster"] = labels

        # Summary 4×4
        summary = (
            rfm_with_cluster
            .groupby("Cluster")
            .agg(Recency=("Recency", "mean"),
                 Frequency=("Frequency", "mean"),
                 Monetary=("Monetary", "mean"),
                 CustomerCount=("Cluster", "count"))
            .round(2)
            .reset_index()
        )

        # แสดงตารางสรุป (คงอยู่จาก session_state แม้ rerun)
        st.write("**Cluster Summary**")
        st.dataframe(summary, use_container_width=True)

        # Silhouette (ถ้าจำนวนคลัสเตอร์เหมาะสม)
        if len(set(labels)) > 1 and len(labels) > len(set(labels)):
            try:
                sil = silhouette_score(rfm_scaled, labels)
                st.caption(f"Silhouette score: {sil:.3f}")
            except Exception:
                pass

        # เตรียมข้อมูลส่งเข้า LLM: list[dict] (เล็กมาก)
        summary_rows = summary.to_dict(orient="records")

        # แปลงเป็นคีย์แคชที่ hashable (ไม่ยิงซ้ำเมื่อข้อมูลเท่าเดิม)
        summary_key = tuple(tuple(sorted(d.items())) for d in summary_rows)

        with st.spinner("กำลังอธิบายผลด้วย Gemini 2.0 Flash..."):
            explanation = explain_cached("gemini-2.0-flash", summary_key, summary_rows)
            st.session_state["explanation"] = explanation

    # ========= 4) แสดงผล AI ที่คงอยู่แม้ rerun =========
    if "explanation" in st.session_state:
        st.subheader("🤖 Gemini Analysis of Clusters")
        st.write(st.session_state["explanation"])

                                height=600)
    st.plotly_chart(silhouette_fig)
