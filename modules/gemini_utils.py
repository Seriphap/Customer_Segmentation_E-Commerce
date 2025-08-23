# modules/gemini_utils.py
import time
import random
import streamlit as st
from google.api_core import exceptions as google_exceptions

def call_gemini_with_backoff(model, prompt, max_retries=5, base=2, max_sleep=60):
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
            time.sleep(sleep)

    # If still fails after retries → show Streamlit error
    if isinstance(last_exception, google_exceptions.ResourceExhausted):
        st.error("⚠️ Gemini API quota is exhausted. Please try again later.")
    elif isinstance(last_exception, google_exceptions.DeadlineExceeded):
        st.error("⏳ Gemini API request timed out. Please retry.")
    elif isinstance(last_exception, google_exceptions.ServiceUnavailable):
        st.error("🚧 Gemini API service is currently unavailable. Try again later.")

    raise last_exception

