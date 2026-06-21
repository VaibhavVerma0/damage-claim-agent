"""
Thin wrapper around the Gemini API: handles forced-JSON structured output,
retries on rate-limit/transient errors, simple pacing to stay under free-tier
quotas, and tracks usage stats so the evaluation report numbers are real.
"""

import time

import google.generativeai as genai

import config


class UsageStats:
    """Accumulates everything the evaluation report needs to cite."""

    def __init__(self):
        self.call_count = 0
        self.failed_calls = 0
        self.retried_calls = 0
        self.images_processed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    @property
    def elapsed_seconds(self):
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def as_dict(self):
        return {
            "call_count": self.call_count,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "images_processed": self.images_processed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class GeminiClient:
    def __init__(self, api_key=None, model_name=config.MODEL_NAME):
        api_key = api_key or config.get_api_key()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.stats = UsageStats()
        self._last_call_time = 0.0

    def _pace(self):
        elapsed = time.time() - self._last_call_time
        wait = config.MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.time()

    def generate_json(self, prompt_text, images, response_schema):
        """
        images: list of (mime_type, base64_data) tuples.
        response_schema: a Gemini-compatible JSON schema dict.
        Returns the parsed JSON dict on success, or None after exhausting retries.
        """
        parts = [{"text": prompt_text}]
        for mime_type, b64_data in images:
            parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})

        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }

        last_error = None
        for attempt in range(config.MAX_RETRIES):
            self._pace()
            try:
                response = self.model.generate_content(
                    parts, generation_config=generation_config
                )
                self.stats.call_count += 1
                self.stats.images_processed += len(images)

                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    self.stats.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                    self.stats.output_tokens += getattr(
                        usage, "candidates_token_count", 0
                    ) or 0

                import json as _json
                return _json.loads(response.text)

            except Exception as e:  # noqa: BLE001 - want to retry on anything transient
                last_error = e
                self.stats.retried_calls += 1
                time.sleep(config.RETRY_BACKOFF_SECONDS * (attempt + 1))

        self.stats.failed_calls += 1
        print(f"  [warn] generation failed after {config.MAX_RETRIES} attempts: {last_error}")
        return None
