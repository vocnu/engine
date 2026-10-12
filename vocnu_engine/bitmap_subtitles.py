"""Bitmap subtitle tracks (Blu-ray PGS) read with OCR, so they can be evidence and song/sign cues too.

ai-voice-over v4 read the English PGS subtitles of every episode with Tesseract (ai_job_sources.extract, frames of
the subtitle drawn on black at 1 fps). Vocnu decodes the PGS stream itself instead: every display set carries its
exact presentation time and the next one ends it, so cues keep the authored timing, and each subtitle bitmap is
turned into dark text on white and read once. Where a line sits on screen is kept: songs and signs are usually
placed away from the dialogue line.
"""
import concurrent.futures as cf
import re
import shutil
import struct
import subprocess

import numpy as np

from . import media

PDS, ODS, PCS, WDS, END = 0x14, 0x15, 0x16, 0x17, 0x80
BITMAP_CODECS = {'hdmv_pgs_subtitle', 'pgssub'}
# Track language tag → Tesseract language (the Modal image installs these).
TESSERACT = {
    'eng': 'eng', 'en': 'eng', 'lit': 'lit', 'lt': 'lit', 'lav': 'lav', 'est': 'est', 'pol': 'pol', 'ger': 'deu',
    'deu': 'deu', 'de': 'deu', 'fre': 'fra', 'fra': 'fra', 'fr': 'fra', 'spa': 'spa', 'es': 'spa', 'por': 'por',
    'pt': 'por', 'ita': 'ita', 'it': 'ita', 'dut': 'nld', 'nld': 'nld', 'nl': 'nld', 'swe': 'swe', 'sv': 'swe',
    'dan': 'dan', 'da': 'dan', 'nor': 'nor', 'nob': 'nor', 'nb': 'nor', 'fin': 'fin', 'fi': 'fin', 'cze': 'ces',
    'ces': 'ces', 'cs': 'ces', 'slo': 'slk', 'slk': 'slk', 'sk': 'slk', 'slv': 'slv', 'sl': 'slv', 'hrv': 'hrv',
    'hr': 'hrv', 'bul': 'bul', 'bg': 'bul', 'rum': 'ron', 'ron': 'ron', 'ro': 'ron', 'hun': 'hun', 'hu': 'hun',
    'gre': 'ell', 'ell': 'ell', 'el': 'ell', 'ukr': 'ukr', 'uk': 'ukr', 'rus': 'rus', 'ru': 'rus', 'tur': 'tur',
    'tr': 'tur', 'ara': 'ara', 'ar': 'ara', 'hin': 'hin', 'hi': 'hin', 'ind': 'ind', 'id': 'ind', 'may': 'msa',
    'msa': 'msa', 'ms': 'msa', 'vie': 'vie', 'vi': 'vie', 'tha': 'tha', 'th': 'tha', 'jpn': 'jpn', 'ja': 'jpn',
    'kor': 'kor', 'ko': 'kor', 'chi': 'chi_sim', 'zho': 'chi_sim', 'zh': 'chi_sim',
}
OCR_PARALLEL = 4


def decode_rle(data, width, height):
    """Palette indices of one PGS object (run-length coded rows)."""
    out = np.zeros((height, width), np.uint8)
    x = y = i = 0
    n = len(data)
    while i < n and y < height:
        byte = data[i]
        i += 1
        if byte:
            if x < width:
                out[y, x] = byte
            x += 1
            continue
        if i >= n:
            break
        flag = data[i]
        i += 1
        if flag == 0:
            x, y = 0, y + 1
            continue
        kind = flag >> 6
        if kind == 0:
            length, color = flag & 0x3F, 0
        elif kind == 1:
            length, color = ((flag & 0x3F) << 8) | data[i], 0
            i += 1
        elif kind == 2:
            length, color = flag & 0x3F, data[i]
            i += 1
        else:
            length, color = ((flag & 0x3F) << 8) | data[i], data[i + 1]
            i += 2
        if color:
            out[y, x:x + length] = color
        x += length
    return out


def display_sets(data):
    """[{'start': s, 'objects': [{'x', 'y', 'image'}], 'height'}] of a PGS (.sup) stream; a set without objects
    clears the screen. `image` is brightness 0-255 of the subtitle composited on black."""
    sets, palettes, objects, current = [], {}, {}, None
    pos = 0
    while pos + 13 <= len(data):
        if data[pos:pos + 2] != b'PG':
            raise ValueError(f'not a PGS stream at byte {pos}')
        pts = struct.unpack('>I', data[pos + 2:pos + 6])[0] / 90000.0
        kind, size = data[pos + 10], struct.unpack('>H', data[pos + 11:pos + 13])[0]
        body = data[pos + 13:pos + 13 + size]
        pos += 13 + size
        if kind == PCS and len(body) >= 11:
            width, height = struct.unpack('>HH', body[0:4])
            state, palette_id, count = body[7], body[9], body[10]
            if state & 0x80:  # epoch start: earlier objects and palettes are gone
                palettes, objects = {}, {}
            placed, at = [], 11
            for _ in range(count):
                object_id, _window, cropped, x, y = struct.unpack('>HBBHH', body[at:at + 8])
                at += 16 if cropped & 0x40 else 8
                placed.append((object_id, x, y))
            current = {'start': pts, 'width': width, 'height': height, 'palette': palette_id, 'placed': placed}
        elif kind == PDS and len(body) >= 2:
            palette = palettes.setdefault(body[0], np.zeros((256, 4), np.float32))
            for at in range(2, len(body) - 4, 5):
                palette[body[at]] = (body[at + 1], body[at + 2], body[at + 3], body[at + 4])
        elif kind == ODS and len(body) >= 4:
            object_id, sequence = struct.unpack('>H', body[0:2])[0], body[3]
            if sequence & 0x80:
                width, height = struct.unpack('>HH', body[7:11])
                objects[object_id] = {'width': width, 'height': height, 'rle': bytearray(body[11:]), 'image': None}
            elif object_id in objects:
                objects[object_id]['rle'] += body[4:]
                objects[object_id]['image'] = None
        elif kind == END and current is not None:
            palette = palettes.get(current['palette'])
            shown = []
            for object_id, x, y in current['placed']:
                found = objects.get(object_id)
                if found is None or palette is None:
                    continue
                indices = decode_rle(found['rle'], found['width'], found['height'])
                image = (palette[indices, 0] * palette[indices, 3] / 255.0).astype(np.uint8)
                shown.append({'x': x, 'y': y, 'image': image})
            sets.append({'start': round(current['start'], 3), 'height': current['height'], 'objects': shown})
            current = None
    return sets


def _pgm(image, pad=12):
    """Dark text on white, padded, as PGM bytes for Tesseract (the subtitle fill is the bright part)."""
    canvas = np.full((image.shape[0] + 2 * pad, image.shape[1] + 2 * pad), 255, np.uint8)
    canvas[pad:pad + image.shape[0], pad:pad + image.shape[1]] = 255 - image
    return b'P5\n%d %d\n255\n' % (canvas.shape[1], canvas.shape[0]) + canvas.tobytes()


# Tesseract reads a subtitle font's capital I as a bar or slash when it stands alone ("| see", "/ offer").
_LONE_I = re.compile(r'(?<![\w|/])[|/](?=\s+\w|\s*$)|(?<=\s)[|/](?=\s)')
_LATIN = frozenset({'eng', 'lit', 'lav', 'est', 'pol', 'deu', 'fra', 'spa', 'por', 'ita', 'nld', 'swe', 'dan', 'nor',
                    'fin', 'ces', 'slk', 'slv', 'hrv', 'ron', 'hun', 'tur', 'ind', 'msa', 'vie'})


def ocr(image, language):
    """Text of one subtitle bitmap; one line per text row."""
    done = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', language, '--psm', '6'], input=_pgm(image),
                          capture_output=True, timeout=60, check=False)
    if done.returncode != 0:
        raise media.MediaError(done.stderr.decode('utf-8', 'replace')[-300:])
    lines = [line.strip() for line in done.stdout.decode('utf-8', 'replace').splitlines() if line.strip()]
    if language in _LATIN:
        lines = [_LONE_I.sub('I', line) for line in lines]
    return '\n'.join(lines)


def available_languages():
    if not shutil.which('tesseract'):
        return set()
    done = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True, timeout=30, check=False)
    return {line.strip() for line in done.stdout.splitlines()[1:] if line.strip()}


def cues(sets, language, *, parallel=OCR_PARALLEL):
    """[(start, end, text, place)] from display sets: each shown set lasts until the next set; `place` is 'top'
    when every line sits in the upper half of the picture, else 'bottom'. Identical consecutive texts merge."""
    shown = [(k, s) for k, s in enumerate(sets) if s['objects']]

    def read(item):
        k, s = item
        parts = sorted(s['objects'], key=lambda o: (o['y'], o['x']))
        return k, '\n'.join(text for text in (ocr(o['image'], language) for o in parts) if text)

    with cf.ThreadPoolExecutor(parallel) as pool:
        texts = dict(pool.map(read, shown))
    found = []
    for k, s in shown:
        end = sets[k + 1]['start'] if k + 1 < len(sets) else s['start'] + 5.0
        text = ' '.join(texts.get(k, '').split())
        if not text or end <= s['start']:
            continue
        top = all(o['y'] + o['image'].shape[0] <= s['height'] / 2 for o in s['objects'])
        place = 'top' if top else 'bottom'
        if found and found[-1][2] == text and abs(found[-1][1] - s['start']) < 0.05:
            found[-1] = (found[-1][0], round(end, 3), text, place)
        else:
            found.append((s['start'], round(end, 3), text, place))
    return found


def extract(source, track, language):
    """Cues of one PGS stream of `source` (see `cues`)."""
    data = media.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', source, '-map', f"0:{track['index']}", '-c', 'copy',
                      '-f', 'sup', 'pipe:1'], timeout=300)
    return cues(display_sets(data), language)
