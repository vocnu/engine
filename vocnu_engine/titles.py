"""The shared glossary of known titles, kept by Rails (Vocnu's worker contract).

For a film, TV series or anime a job identified, Rails returns the renderings earlier jobs decided for its names
and recurring terms, and stores those this job is the first to decide. Best effort: when Rails cannot be reached,
the job keeps its own glossary.

Local runs have no Rails callback: they read the public names library instead (backend ADR 0016, open data under
CC BY-SA 4.0) and never write to it.
"""
import json
import logging
import re
import unicodedata
from urllib.parse import quote, urlsplit

from . import http, signing
from .config import secret
from .errors import ProviderError

PATH = '/api/v1/worker/glossary'
NAMES_API = 'https://app.vocnu.com/api/v1/names'

log = logging.getLogger('vocnu.titles')


def endpoint(callback_url):
    """The glossary URL on the Rails app that receives the job's events; None without an https callback."""
    parts = urlsplit(callback_url or '')
    if parts.scheme != 'https' or not parts.netloc:
        return None
    return f'https://{parts.netloc}{PATH}'


def key_for(name):
    """The library's key of a title name, as Rails derives it (MediaTitles::Glossary.key_for)."""
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(name)) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:80].rstrip('-')


def library(names_api, title, language, *, session=None):
    """The public names library's renderings [{'source', 'target'}] for a title; None when it has none."""
    key = key_for(title or '')
    if not names_api or not key:
        return None
    url = f"{names_api.rstrip('/')}/{quote(key)}"
    try:
        data = http.request('GET', url, provider='Vocnu names library', timeout=20, attempts=2,
                            session=session).json()
    except (ProviderError, ValueError) as error:   # a title the library does not have answers 404
        log.info('names library: %s', error)
        return None
    entries = (((data.get('title') or {}).get('terms') or {}).get(language) or []) if isinstance(data, dict) else []
    return [{'source': str(e['source']), 'target': str(e['rendering'])} for e in entries
            if isinstance(e, dict) and e.get('source') and e.get('rendering')]


def sync(callback_url, title, language, terms, job_id, *, names_api=None, session=None):
    """Every decided [{'source', 'target'}] for the title after adding `terms`; None when unavailable.

    Without a Rails callback (local runs) the public library at `names_api` is read, never written: storing
    `terms` there returns None, so the job keeps its own glossary.
    """
    url = endpoint(callback_url)
    if not title:
        return None
    if not url:
        return None if terms else library(names_api, title, language, session=session)
    body = json.dumps({'title': title, 'language': language, 'terms': list(terms), 'job_id': job_id},
                      ensure_ascii=False).encode()
    headers = {**signing.sign(secret('WORKER_SHARED_SECRET'), body), 'Content-Type': 'application/json'}
    try:
        data = http.request('POST', url, provider='Vocnu', data=body, headers=headers, timeout=30, attempts=3,
                            session=session).json()
    except (ProviderError, ValueError) as error:
        log.warning('shared glossary unavailable: %s', error)
        return None
    return [{'source': str(t['source']), 'target': str(t['target'])} for t in data.get('terms') or []
            if isinstance(t, dict) and t.get('source') and t.get('target')]
