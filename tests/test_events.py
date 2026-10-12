import json
import threading
import time

from vocnu_engine import signing
from vocnu_engine.events import EventSink


class FakeRails:
    """Records signed posts; answers with queued status codes (default 200)."""

    def __init__(self, statuses=()):
        self.statuses = list(statuses)
        self.posts = []
        self.lock = threading.Lock()

    def __call__(self, url, body, headers):
        signing.verify('secret', body, headers['X-Vocnu-Timestamp'], headers['X-Vocnu-Signature'])
        with self.lock:
            self.posts.append(json.loads(body))
            return self.statuses.pop(0) if self.statuses else 200


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_progress_events_are_signed_and_sequenced():
    rails = FakeRails()
    sink = EventSink('job-1234567', 'https://rails/events', 'secret', post=rails)
    assert sink.progress('downloading', 0.0)
    assert sink.progress('transcribing', 0.1)
    assert [p['sequence'] for p in rails.posts] == [1, 2]
    assert rails.posts[1] == {'job_id': 'job-1234567', 'sequence': 2, 'event': 'progress',
                              'stage': 'transcribing', 'progress': 0.1}


def test_terminal_event_retries_until_delivered():
    rails = FakeRails(statuses=[500, 503, 200])
    clock = Clock()
    sink = EventSink('job-1234567', 'https://rails/events', 'secret', post=rails, sleep=clock.sleep, clock=clock)
    assert sink.completed({'duration_seconds': 3.0, 'outputs': [], 'receipt': {}})
    assert len(rails.posts) == 3 and rails.posts[-1]['event'] == 'completed'
    assert sink.terminal_sent


def test_terminal_event_gives_up_after_the_retry_window():
    rails = FakeRails(statuses=[500] * 1000)
    clock = Clock()
    sink = EventSink('job-1234567', 'https://rails/events', 'secret', post=rails, sleep=clock.sleep, clock=clock,
                     terminal_retry_seconds=120)
    assert not sink.failed({'code': 'internal_error', 'message': 'x', 'retryable': True}, {})
    assert clock.now <= 120 and len(rails.posts) > 3


def test_heartbeat_resends_the_current_stage_with_new_sequence():
    rails = FakeRails()
    sink = EventSink('job-1234567', 'https://rails/events', 'secret', post=rails, heartbeat_seconds=0.05)
    sink.start_heartbeat()
    sink.progress('synthesizing', 0.5)
    time.sleep(0.4)
    sink.stop_heartbeat()
    beats = [p for p in rails.posts if p['stage'] == 'synthesizing']
    assert len(beats) >= 2
    assert all(b['progress'] == 0.5 for b in beats)
    sequences = [p['sequence'] for p in rails.posts]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)


def test_progress_never_goes_backwards():
    rails = FakeRails()
    sink = EventSink('job-1234567', 'https://rails/events', 'secret', post=rails)
    sink.progress('translating', 0.4)
    sink.progress('translating', 0.3)
    assert rails.posts[-1]['progress'] == 0.4
