import shutil
import struct
import subprocess

import numpy as np
import pytest

from vocnu_engine import bitmap_subtitles

# Two rows of four pixels: [5, 5, 0, 0] and [0, 7, 7, 7] (run-length coded as PGS writes them).
RLE = bytes([0x00, 0x82, 0x05, 0x00, 0x02, 0x00, 0x00,
             0x00, 0x01, 0x00, 0x83, 0x07, 0x00, 0x00])


def segment(kind, pts, body):
    return b'PG' + struct.pack('>IIBH', round(pts * 90000), 0, kind, len(body)) + body


def composition(pts, objects, number=0, state=0x80, height=1080):
    body = struct.pack('>HHBHBBBB', 1920, height, 0x10, number, state, 0, 0, len(objects))
    for object_id, x, y in objects:
        body += struct.pack('>HBBHH', object_id, 0, 0, x, y)
    return segment(bitmap_subtitles.PCS, pts, body)


def display_set(pts, y):
    palette = struct.pack('>BB', 0, 0) + bytes([5, 235, 128, 128, 255, 7, 16, 128, 128, 255])
    shown = struct.pack('>HBB', 0, 0, 0xC0) + (len(RLE) + 4).to_bytes(3, 'big') + struct.pack('>HH', 4, 2) + RLE
    return (composition(pts, [(0, 100, y)]) + segment(bitmap_subtitles.PDS, pts, palette)
            + segment(bitmap_subtitles.ODS, pts, shown) + segment(bitmap_subtitles.END, pts, b''))


def clear(pts):
    return composition(pts, [], state=0) + segment(bitmap_subtitles.END, pts, b'')


def test_run_length_rows_decode_to_palette_indices():
    assert bitmap_subtitles.decode_rle(RLE, 4, 2).tolist() == [[5, 5, 0, 0], [0, 7, 7, 7]]


def test_display_sets_keep_exact_times_and_the_subtitle_brightness():
    sets = bitmap_subtitles.display_sets(display_set(1.0, 900) + clear(3.5))
    assert [s['start'] for s in sets] == [1.0, 3.5]
    image = sets[0]['objects'][0]['image']
    # The bright fill (Y 235) stays bright, the dark outline (Y 16) and transparent pixels are dark.
    assert image.tolist() == [[235, 235, 0, 0], [0, 16, 16, 16]]
    assert sets[1]['objects'] == []


def test_cues_last_until_the_next_set_and_know_their_place(monkeypatch):
    monkeypatch.setattr(bitmap_subtitles, 'ocr', lambda image, language: 'Raise your eyes')
    data = display_set(2.0, 80) + clear(5.75) + display_set(7.0, 950) + clear(8.5)
    assert bitmap_subtitles.cues(bitmap_subtitles.display_sets(data), 'eng') == [
        (2.0, 5.75, 'Raise your eyes', 'top'), (7.0, 8.5, 'Raise your eyes', 'bottom')]


def test_a_stream_that_is_not_pgs_is_refused():
    with pytest.raises(ValueError):
        bitmap_subtitles.display_sets(b'not a subtitle stream')


@pytest.mark.skipif(not shutil.which('tesseract') or 'eng' not in bitmap_subtitles.available_languages(),
                    reason='needs Tesseract with English')
def test_ocr_reads_a_subtitle_drawn_light_on_dark():
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=640x90',
                          '-vf', "drawtext=text='I see shooting stars':fontcolor=white:fontsize=48:x=20:y=18",
                          '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
                         capture_output=True, check=True).stdout
    image = np.frombuffer(raw, np.uint8).reshape(90, 640)
    assert bitmap_subtitles.ocr(image, 'eng') == 'I see shooting stars'
