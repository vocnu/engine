"""HTTP with bounded retries for paid providers. Never logs request bodies, headers or keys."""
import random
import time

import requests

from .errors import ProviderError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def request(method, url, *, provider, attempts=5, timeout=120, session=None, sleep=time.sleep,
            ambiguous_retries=None, **kwargs):
    """Response for a 2xx answer; ProviderError after `attempts` or on a non-retryable status.

    `ambiguous_retries` is for paid requests whose failure may still be billed (ai-voice-over v4 provider_retry: the
    first unknown outcome plus one controlled replay). A 5xx or a timeout can come after the provider did the work,
    so those are retried at most this many times; a 429 is refused before any work (not billed) and keeps using
    `attempts`. None (the default) retries every retryable failure alike.
    """
    client = session or requests
    delay = 2.0
    ambiguous = 0
    for attempt in range(1, attempts + 1):
        try:
            response = client.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException:
            response = None
        if response is not None and response.ok:
            return response
        status = response.status_code if response is not None else None
        if status is not None and status not in RETRY_STATUS:
            raise ProviderError(f'{provider} rejected the request (HTTP {status}).', status)
        if ambiguous_retries is not None and status != 429:
            ambiguous += 1
            if ambiguous > ambiguous_retries:
                break
        if attempt == attempts:
            break
        retry_after = response.headers.get('Retry-After') if response is not None else None
        wait = float(retry_after) if retry_after and retry_after.replace('.', '', 1).isdigit() else delay
        sleep(min(wait, 60.0) + random.uniform(0, 0.5))
        delay = min(delay * 2, 30.0)
    raise ProviderError(f'{provider} is unavailable right now (HTTP {status or "timeout"}).', status)
