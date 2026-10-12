"""Turn a contract `source` into a local media file: R2 uploads or local files (CLI).

Link imports (YouTube) were removed on 2026-10-08 (root ADR 0006); any other source kind is refused.
"""
from pathlib import Path
import re

from . import media
from .errors import WorkerError
from .storage import check_upload_key


def probe(source, store=None):
    """Contract probe for an upload key (via a presigned URL) or a local file (CLI)."""
    kind = (source or {}).get('kind')
    if kind == 'upload':
        key = check_upload_key(source.get('key'))
        if store.size(key) is None:
            raise WorkerError('not_found', 'The uploaded file was not found. It may have expired; upload it again.')
        info = media.inspect(store.presign_get(key))
        info['title'] = Path(key).stem
        return media.public_probe(info)
    if kind == 'file':
        return media.public_probe(media.inspect(source['path']))
    raise WorkerError('not_found', 'Unknown source kind.')


def fetch(source, directory, store=None, avoid_languages=(), prefer_languages=()):
    """Download the source into `directory`; return (local path, inspect() facts).

    `avoid_languages`: audio tracks in these languages (the target's) are not taken as the source speech;
    `prefer_languages`: the source language the customer named, if any.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kind = (source or {}).get('kind')
    title = None
    if kind == 'upload':
        key = check_upload_key(source.get('key'))
        suffix = Path(key).suffix.lower()[:8] if re.fullmatch(r'\.[a-z0-9]{1,7}', Path(key).suffix.lower()) else ''
        path = store.download(key, directory / f'source{suffix}')
        title = Path(key).stem
    elif kind == 'file':
        path = Path(source['path'])
        if not path.is_file():
            raise WorkerError('not_found', 'The source file was not found.')
    else:
        raise WorkerError('not_found', 'Unknown source kind.')
    info = media.inspect(path, avoid_languages=avoid_languages, prefer_languages=prefer_languages)
    info['title'] = title or info.get('title')
    return Path(path), info
