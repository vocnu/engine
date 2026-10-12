"""Signed job events to Rails (worker-contract.md), with retries and a progress heartbeat.

Terminal events (`completed`, `failed`) are retried with backoff for up to 15 minutes; progress is
best effort. While a job runs, a heartbeat resends the current stage at least every 2 minutes so
Rails' 20-minute watchdog never mistakes a long stage for a dead job.
"""
import json
import threading
import time

import requests

from . import signing

STAGES = ('downloading', 'transcribing', 'translating', 'synthesizing', 'mixing', 'rendering', 'uploading')
HEARTBEAT_SECONDS = 90
TERMINAL_RETRY_SECONDS = 15 * 60


class EventSink:
    """Posts events for one job. `post` is injectable for tests."""

    def __init__(self, job_id, callback_url, secret, *, post=None, sleep=time.sleep, clock=time.monotonic,
                 heartbeat_seconds=HEARTBEAT_SECONDS, terminal_retry_seconds=TERMINAL_RETRY_SECONDS,
                 start_sequence=0):
        self.job_id = job_id
        self.callback_url = callback_url
        self.secret = secret
        self._post = post or self._http_post
        self._sleep = sleep
        self._clock = clock
        self.heartbeat_seconds = heartbeat_seconds
        self.terminal_retry_seconds = terminal_retry_seconds
        self._sequence = start_sequence  # a cancel sink starts far above any running job's sequence
        self._lock = threading.Lock()
        self._stage = None
        self._progress = 0.0
        self._last_sent = None
        self._stop = threading.Event()
        self._thread = None
        self.terminal_sent = False

    # Transport -----------------------------------------------------------------------------------
    @staticmethod
    def _http_post(url, body, headers):
        response = requests.post(url, data=body, headers=headers, timeout=20)
        return response.status_code

    def _send(self, payload, *, attempts_deadline):
        """POST until a 2xx answer or the deadline; returns True when delivered."""
        body = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode()
        delay, deadline = 2.0, self._clock() + attempts_deadline
        while True:
            headers = {'Content-Type': 'application/json', **signing.sign(self.secret, body)}
            try:
                status = self._post(self.callback_url, body, headers)
            except requests.RequestException:
                status = None
            if status is not None and 200 <= status < 300:
                return True
            if self._clock() + delay > deadline:
                return False
            self._sleep(delay)
            delay = min(delay * 2, 60.0)

    def _next(self, **fields):
        with self._lock:
            self._sequence += 1
            return {'job_id': self.job_id, 'sequence': self._sequence, **fields}

    # Events --------------------------------------------------------------------------------------
    def progress(self, stage, progress):
        if stage not in STAGES:
            raise ValueError(f'Unknown stage {stage!r}')
        with self._lock:
            self._stage, self._progress = stage, max(self._progress, min(1.0, max(0.0, float(progress))))
            current = self._progress
        delivered = self._send(self._next(event='progress', stage=stage, progress=round(current, 4)),
                               attempts_deadline=20)
        with self._lock:
            self._last_sent = self._clock() if delivered else self._last_sent
        return delivered

    def completed(self, payload):
        return self._terminal(event='completed', **payload)

    def failed(self, error, receipt):
        return self._terminal(event='failed', error=error, receipt=receipt)

    def _terminal(self, **fields):
        self.stop_heartbeat()
        delivered = self._send(self._next(**fields), attempts_deadline=self.terminal_retry_seconds)
        self.terminal_sent = delivered
        return delivered

    # Heartbeat -----------------------------------------------------------------------------------
    def start_heartbeat(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._beat, name=f'heartbeat-{self.job_id}', daemon=True)
            self._thread.start()

    def stop_heartbeat(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _beat(self):
        while not self._stop.wait(min(15.0, self.heartbeat_seconds)):
            with self._lock:
                stage, due = self._stage, (self._last_sent is None
                                           or self._clock() - self._last_sent >= self.heartbeat_seconds)
            if stage and due:
                self.progress(stage, self._progress)


class NullSink:
    """For local CLI runs: prints progress instead of calling Rails."""

    def __init__(self, echo=print):
        self.echo = echo
        self.terminal_sent = False

    def progress(self, stage, progress):
        self.echo(f'[{progress:5.1%}] {stage}')
        return True

    def completed(self, payload):
        self.terminal_sent = True
        return True

    def failed(self, error, receipt):
        self.terminal_sent = True
        self.echo(f"failed: {error['code']}: {error['message']}")
        return True

    def start_heartbeat(self):
        pass

    def stop_heartbeat(self):
        pass
