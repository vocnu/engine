"""Failure codes from the worker contract, carried by one exception type."""

PROBE_CODES = ('not_found', 'unsupported_media', 'no_audio', 'too_long')
JOB_CODES = PROBE_CODES + ('duration_mismatch', 'budget_exceeded', 'provider_error', 'internal_error', 'cancelled')
RETRYABLE = frozenset({'provider_error', 'internal_error'})


class WorkerError(Exception):
    """An expected failure with a contract code and a message safe to show users."""

    def __init__(self, code, message):
        if code not in JOB_CODES:
            raise ValueError(f'Unknown failure code {code!r}')
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def retryable(self):
        return self.code in RETRYABLE

    def as_dict(self):
        return {'code': self.code, 'message': self.message, 'retryable': self.retryable}


class ProviderError(WorkerError):
    """A paid provider (OpenAI, ElevenLabs) failed after retries."""

    def __init__(self, message, status=None):
        super().__init__('provider_error', message)
        self.status = status
