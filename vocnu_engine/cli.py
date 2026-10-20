"""Run Vocnu's engine on this computer: speech recognition on its GPU (or CPU), translation with your OpenAI key and
voices with your ElevenLabs key.

    vocnu-engine translate <file> --to lt [--output video|audio|subtitles] [--voice female|male] [--budget 5]
    vocnu-engine probe <file>
    vocnu-engine languages
    vocnu-engine process <file> --target lt [--products all] [--voice female]
        [--source-language auto] [--budget 1.00] [--out work/out] [--work work/jobs]

In the Vocnu repository the same commands run as `uv run python -m vocnu_engine.cli ...`. `translate` saves the
files next to the media (or in --out); `process` is the job contract as Vocnu runs it, with outputs in
<out>/jobs/<job_id>/ and the receipt in receipt.json. Keys come from the environment or the private files in
~/.config/vocnu (see config.py). Known films and series take their names from the public Vocnu names library.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import uuid

from . import asr, pipeline, render, sources, titles
from .errors import WorkerError
from .events import NullSink
from .languages import LANGUAGES
from .media import AUDIO_CODECS
from .storage import LocalStore

OUTPUTS = ('video', 'audio', 'subtitles')
# Demo runs cost $0.05-0.10 per minute of speech (translation and voices); this leaves room for dense dialogue.
BUDGET_PER_MINUTE = 0.15


def _source(target):
    return {'kind': 'file', 'path': str(Path(target).resolve())}


def _products(value):
    names = pipeline.PRODUCTS if value == 'all' else [p.strip() for p in value.split(',') if p.strip()]
    unknown = set(names) - set(pipeline.PRODUCTS)
    if unknown:
        raise argparse.ArgumentTypeError(f'unknown products: {", ".join(sorted(unknown))}')
    return {name: True for name in names}


def _video_formats(value):
    """`mkv` or several formats at once, `mp4,mkv`: each comes as its own file."""
    names = list(dict.fromkeys(name.strip().lower() for name in value.split(',') if name.strip()))
    if not names or any(name not in render.VIDEO_FORMATS for name in names):
        raise argparse.ArgumentTypeError(f'choose from {", ".join(render.VIDEO_FORMATS)}, separated by commas')
    return names


def products_for(output, burn_subtitles=False):
    """The job products of an output, as the Vocnu app and CLI choose them: every output includes subtitles."""
    return {'subtitles': True, 'voiceover_audio': output != 'subtitles', 'voiceover_video': output == 'video',
            'burned_subtitles': output == 'video' and burn_subtitles}


class FolderStore:
    """Saves each output in one folder under its own file name."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def upload(self, path, key, content_type):
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, self.folder / Path(key).name)
        return key


def _parser():
    parser = argparse.ArgumentParser(prog='vocnu-engine', description='AI voice-over and subtitles for your videos.')
    commands = parser.add_subparsers(dest='command', required=True)
    translate = commands.add_parser('translate', help='translate a video or audio file into another language')
    translate.add_argument('source', help='the video or audio file')
    translate.add_argument('--to', required=True, help='target language code, e.g. lt, de, es (see `languages`)')
    translate.add_argument('--from', dest='source_language', default='auto', help='spoken language (default: detected)')
    translate.add_argument('--output', choices=OUTPUTS, help='video (default for videos), audio or subtitles only')
    translate.add_argument('--voice', default='female', choices=('female', 'male'))
    translate.add_argument('--format', dest='video_formats', type=_video_formats,
                           help=f"video file format ({', '.join(render.VIDEO_FORMATS)}), or several separated by "
                                "commas, e.g. mp4,mkv (default: the source's, else mp4)")
    translate.add_argument('--audio-format', default='m4a', choices=tuple(AUDIO_CODECS))
    translate.add_argument('--burn-subtitles', action='store_true', help='burn the subtitles into the picture')
    translate.add_argument('--budget', type=float, default=None,
                           help='hard limit in USD on what your OpenAI and ElevenLabs keys may spend '
                                f'(default: ${BUDGET_PER_MINUTE:.2f} per minute of media)')
    translate.add_argument('--out', help='folder for the results (default: next to the file)')
    translate.add_argument('--work', default=None,
                           help='folder for intermediate files, kept afterwards (default: a cache folder that is '
                                'removed after success and reused by a retry, so paid speech is not bought twice)')
    translate.add_argument('--names-api', default=os.environ.get('VOCNU_NAMES_API', titles.NAMES_API),
                           help='public names library to read known titles from')
    translate.add_argument('--no-names-library', action='store_true', help='do not read the public names library')
    probe = commands.add_parser('probe', help='duration and tracks of a media file')
    probe.add_argument('source')
    commands.add_parser('languages', help='languages the engine translates into')
    run = commands.add_parser('process', help='run a whole job as Vocnu does')
    run.add_argument('source')
    run.add_argument('--target', required=True)
    run.add_argument('--source-language', default='auto')
    run.add_argument('--products', type=_products, default=_products('all'))
    run.add_argument('--voice', default='female', choices=('female', 'male'))
    run.add_argument('--budget', type=float, default=1.0, help='hard provider spend limit in USD')
    run.add_argument('--out', default='work/out')
    run.add_argument('--work', default='work/jobs')
    run.add_argument('--job-id', default=None)
    return parser


def _probe(source):
    try:
        return sources.probe(source), None
    except WorkerError as error:
        print(json.dumps({'error': error.as_dict()}, indent=2))
        return None, 1


def _job(job_id, source, facts, **fields):
    return {'job_id': job_id, 'source': source, 'max_duration_seconds': round(facts['duration_seconds'] * 1.05 + 1, 1),
            **fields}


def _work_folder(path, target):
    """One cache folder per file and language, so a retry reuses the speech it already paid for."""
    stat = path.stat()
    digest = hashlib.sha256(f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{target}'.encode()).hexdigest()[:16]
    root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'vocnu-engine'
    return root / digest


def _translate(args):
    path = Path(args.source).resolve()
    source = _source(path)
    facts, failed = _probe(source)
    if failed:
        return failed
    output = args.output or ('video' if facts.get('has_video') else 'audio')
    if output == 'video' and not facts.get('has_video'):
        print('This file has no picture: use --output audio or --output subtitles.', file=sys.stderr)
        return 2
    suffix = path.suffix.lower().lstrip('.')
    video_formats = args.video_formats or [suffix if suffix in render.VIDEO_FORMATS else 'mp4']
    budget = args.budget if args.budget is not None else max(1.0, round(
        facts['duration_seconds'] / 60 * BUDGET_PER_MINUTE, 2))
    print(f'Spending at most ${budget:.2f} of your provider credit (change it with --budget).')
    job_id = str(uuid.uuid4())
    job = _job(job_id, source, facts, source_language=args.source_language, target_language=args.to,
               products=products_for(output, args.burn_subtitles), voice=args.voice, video_formats=video_formats,
               audio_format=args.audio_format, budget_usd=budget,
               names_api=None if args.no_names_library else args.names_api)
    folder = Path(args.out) if args.out else path.parent / f'{path.stem} (vocnu {args.to})'
    work = Path(args.work) if args.work else _work_folder(path, args.to)
    result = pipeline.process(job, work, store=FolderStore(folder), sink=NullSink(),
                              transcribe=lambda media, language, hints=(): asr.transcribe(media, language, hints=hints),
                              compute='local')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    if result['event'] != 'completed':
        print(f"Failed: {result['error']['message']} Run the same command again to retry; speech already made is "
              f"reused.", file=sys.stderr)
        return 1
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    for item in result['outputs']:
        print(f"{item['kind']}: {folder / Path(item['key']).name}")
    print(f"Spent ${result['receipt']['total_usd']:.4f} of your ${budget:.2f} budget.")
    return 0


def _process(args):
    source = _source(args.source)
    facts, failed = _probe(source)
    if failed:
        return failed
    job_id = args.job_id or str(uuid.uuid4())
    job = _job(job_id, source, facts, source_language=args.source_language, target_language=args.target,
               products=args.products, voice=args.voice, budget_usd=args.budget)
    out = Path(args.out)
    result = pipeline.process(job, Path(args.work) / job_id, store=LocalStore(out), sink=NullSink(),
                              transcribe=lambda path, language, hints=(): asr.transcribe(path, language, hints=hints),
                              compute='local')
    folder = out / 'jobs' / job_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in result if k != 'receipt'}, indent=2, ensure_ascii=False))
    print(f"spent ${result['receipt']['total_usd']:.4f} of ${result['receipt']['max_usd']:.2f}; outputs in {folder}")
    return 0 if result['event'] == 'completed' else 1


def main(argv=None):
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    if args.command == 'languages':
        for language in LANGUAGES:
            print(f"{language.code:<4} {language.native}{'' if language.voiceover else ' (subtitles only)'}")
        return 0
    if args.command == 'probe':
        facts, failed = _probe(_source(args.source))
        if failed:
            return failed
        print(json.dumps(facts, indent=2, ensure_ascii=False))
        return 0
    if args.command == 'translate':
        return _translate(args)
    return _process(args)


if __name__ == '__main__':
    sys.exit(main())
