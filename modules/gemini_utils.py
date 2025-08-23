# modules/gemini_utils.py
import time
import random
import streamlit as st
from google.api_core import exceptions as google_exceptions

MAX_CALLS_PER_MINUTE = 15
RATE_LIMIT_WINDOW = 60  # seconds

def is_rate_limited():
    now = time.time()
    if "gemini_call_timestamps" not in st.session_state:
        st.session_state.gemini_call_timestamps = []

    # Remove timestamps older than RATE_LIMIT_WINDOW
    st.session_state.gemini_call_timestamps = [
        ts for ts in st.session_state.gemini_call_timestamps if now - ts < RATE_LIMIT_WINDOW
    ]

    if len(st.session_state.gemini_call_timestamps) >= MAX_CALLS_PER_MINUTE:
        return True

    # Record current timestamp
    st.session_state.gemini_call_timestamps.append(now)
    return False

def call_gemini_with_backoff(model, prompt, max_retries=5, base=2, max_sleep=60):
    if is_rate_limited():
        st.warning("🚫 Rate limit exceeded: Please wait before making more requests (max 15/min).")
        return None

    last_exception = None
    for i in range(max_retries):
        try:
            return model.generate_content(prompt)
        except (google_exceptions.ResourceExhausted,
                google_exceptions.DeadlineExceeded,
                google_exceptions.ServiceUnavailable) as e:
            last_exception = e
            sleep = min(max_sleep, base ** i) + random.random()
            time.sleep(sleep)

    if isinstance(last_exception, google_exceptions.ResourceExhausted):
        st.error("⚠️ Gemini API quota is exhausted. Please try again later.")
    elif isinstance(last_exception, google_exceptions.DeadlineExceeded):
        st.error("⏳ Gemini API request timed out. Please retry.")
    elif isinstance(last_exception, google_exceptions.ServiceUnavailable):
        st.error("🚧 Gemini API service is currently unavailable. Try again later.")

    raise last_exception

