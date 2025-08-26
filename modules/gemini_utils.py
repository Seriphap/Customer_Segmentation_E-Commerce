# modules/gemini_utils.py
import time
from google.api_core.exceptions import ResourceExhausted

def call_gemini_with_backoff(model, prompt, max_retries=5, base=2, max_sleep=60):
    for i in range(max_retries):
        try:
            return model.generate_content(prompt.strip())
        except ResourceExhausted as e:
            if i < max_retries - 1:
                sleep = min(max_sleep, base ** i)
                time.sleep(sleep)
            else:
                raise e


