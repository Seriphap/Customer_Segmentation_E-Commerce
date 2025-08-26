import time
import random
import logging
from datetime import datetime
from google.api_core import exceptions as google_exceptions

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _extract_text_from_response(response):
    """
    Best-effort extraction of human-readable text from a GenerateContentResponse-like object.
    Returns a string (possibly empty) but never raises.
    """
    try:
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
    except Exception:
        logger.debug("response.text accessor failed", exc_info=True)

    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            cand = candidates[0]
            content = getattr(cand, "content", None)
            if content:
                parts = []
                for part in content:
                    if hasattr(part, "text"):
                        parts.append(str(part.text))
                    else:
                        parts.append(str(part))
                joined = "\n".join([p for p in parts if p]).strip()
                if joined:
                    return joined
            cand_str = str(cand).strip()
            if cand_str:
                return cand_str
    except Exception:
        logger.debug("candidate/content extraction failed", exc_info=True)

    try:
        output = getattr(response, "output", None)
        if output:
            return str(output).strip()
    except Exception:
        logger.debug("output extraction failed", exc_info=True)

    try:
        return str(response).strip()
    except Exception:
        return ""

def _safe_repr(obj, max_len=1000):
    try:
        s = repr(obj)
        if len(s) > max_len:
            return s[:max_len] + "...[truncated]"
        return s
    except Exception:
        return "<unrepresentable object>"

def call_gemini_with_backoff(
    model,
    prompt,
    max_retries=3,
    base=2,
    max_sleep=60,
    increment_attempt_cb=None,
    log_event_cb=None,
):
    """
    Call model.generate_content with backoff and logging.

    - increment_attempt_cb(): optional callback called once per actual network attempt.
    - log_event_cb(dict): optional callback for logging events. The dict will contain keys:
        'timestamp', 'event', plus event-specific fields like 'attempt', 'error', 'extracted', 'raw_repr'.
    Returns extracted non-empty text on success. Raises on repeated failures.
    """
    def _log(event_name, **kwargs):
        payload = {"timestamp": datetime.utcnow().isoformat() + "Z", "event": event_name}
        payload.update(kwargs)
        logger.info("%s %s", event_name, kwargs)
        if callable(log_event_cb):
            try:
                log_event_cb(payload)
            except Exception:
                logger.exception("log_event_cb failed")

    last_exception = None
    for attempt in range(max_retries):
        _log("attempt_start", attempt=attempt + 1, max_retries=max_retries, prompt_length=len(prompt))
        if increment_attempt_cb:
            try:
                increment_attempt_cb()
            except Exception:
                logger.exception("increment_attempt_cb failed")
        try:
            response = model.generate_content(prompt)
        except (
            google_exceptions.ResourceExhausted,
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
            google_exceptions.GoogleAPICallError,
        ) as e:
            last_exception = e
            _log("transient_exception", attempt=attempt + 1, error=repr(e))
            if attempt < max_retries - 1:
                sleep = min(max_sleep, base ** attempt) + random.uniform(0, 1)
                _log("backoff_sleep", attempt=attempt + 1, sleep_seconds=sleep)
                time.sleep(sleep)
                continue
            else:
                _log("retries_exhausted_transient", attempt=attempt + 1, error=repr(e))
                raise
        except Exception as e:
            _log("unexpected_exception", attempt=attempt + 1, error=repr(e))
            raise

        try:
            extracted = _extract_text_from_response(response)
        except Exception as e:
            extracted = ""
            _log("extraction_failed", attempt=attempt + 1, error=repr(e))

        _log(
            "attempt_response",
            attempt=attempt + 1,
            extracted_snippet=(extracted[:500] + "...") if extracted and len(extracted) > 500 else extracted,
            raw_repr=_safe_repr(response, max_len=1000),
        )

        if extracted:
            _log("success", attempt=attempt + 1, extracted_length=len(extracted))
            return extracted
        else:
            msg = "Empty or invalid response from Gemini."
            last_exception = ValueError(msg)
            _log("empty_response", attempt=attempt + 1)
            if attempt < max_retries - 1:
                sleep = min(max_sleep, base ** attempt) + random.uniform(0, 1)
                _log("backoff_sleep_empty_response", attempt=attempt + 1, sleep_seconds=sleep)
                time.sleep(sleep)
                continue
            else:
                _log("retries_exhausted_empty", attempt=attempt + 1)
                raise last_exception

    if last_exception:
        raise last_exception
    return ""