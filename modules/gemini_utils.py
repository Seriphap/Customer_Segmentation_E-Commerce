# modules/gemini_utils.py
import time
import random
import streamlit as st
from google.api_core import exceptions as google_exceptions

# Free tier: 15 requests/minute/model → ~1 request ทุก 4 วินาที
REQUEST_COOLDOWN = 4  # seconds (per user session)

def call_gemini_with_backoff(model, prompt, max_retries=5, base=2, max_sleep=60):
    """
    Call Gemini API with exponential backoff and per-user rate limiting.
    """

    # --- Rate limiter (per user session) ---
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    now = time.time()
    elapsed = now - st.session_state.last_request_time

    if elapsed < REQUEST_COOLDOWN:
        wait_time = REQUEST_COOLDOWN - int(elapsed)
        st.warning(
            f"⚠️ Please wait {wait_time} seconds before sending another request "
            f"(quota limit: 15 requests/min in free tier)."
        )
        return None  # ไม่ยิง API ถ้ายังไม่ถึงเวลา

    # update เวลาที่ request ล่าสุด
    st.session_state.last_request_time = now

    # --- Retry with backoff ---
    last_exception = None
    for i in range(max_retries):
        try:
            return model.generate_content(prompt)
        except (google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable) as e:
            last_exception = e
            # truncated exponential backoff + jitter
            sleep = min(max_sleep, base ** i) + random.random()
            st.info(f"🔄 Retrying in {sleep:.1f} seconds... (attempt {i+1}/{max_retries})")
            time.sleep(sleep)

    # --- Show error message if all retries failed ---
    if isinstance(last_exception, google_exceptions.ResourceExhausted):
        st.error("⚠️ Gemini API quota is exhausted. Please try again later.")
    elif isinstance(last_exception, google_exceptions.DeadlineExceeded):
        st.error("⏳ Gemini API request timed out. Please retry.")
    elif isinstance(last_exception, google_exceptions.ServiceUnavailable):
        st.error("🚧 Gemini API service is currently unavailable. Try again later.")

    raise last_exception


