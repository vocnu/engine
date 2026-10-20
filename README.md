# Vocnu engine

Open source AI voice-over and subtitles for videos. Give it a film, an episode or a lecture and a target
language; it gives you the video with a narrated voice-over (the original stays audible underneath, the way
Eastern European TV narrators do it), the voice-over audio, and subtitles.

This is the engine behind [Vocnu](https://vocnu.com). The hosted service runs the same code with no setup,
fast GPUs, managed provider keys and support; this repository is for running it yourself and improving it.

```sh
uvx --from git+https://github.com/vocnu/engine vocnu-engine translate film.mkv --to lt
```

## What it does

- **Speech recognition** with faster-whisper (large-v3-turbo) on your GPU or CPU, helped by the subtitle tracks
  the file already has (text, and image subtitles read with Tesseract).
- **Translation** with OpenAI: one analysis of the whole transcript (who speaks, which lines are songs or signs,
  how names are written and declined), chunked translation with context, an independent review, and per-line
  character budgets so every narrator line fits the time it has.
- **Narration** with ElevenLabs voices and character timestamps; lines that run long are retaken, rephrased
  shorter and only then sped up (at most 1.08×). Names the voice says the source language's way
  (Lithuanian "Aliaska" read as English "Alaska") are found by recognising takes again and respelled for the voice
  only; codes such as "C-46" are said as words while the subtitles keep them as written.
- **Mixing**: the original is dipped under each narrator line and mastered; video outputs keep the original
  audio as a second track.
- **Outputs**: MP4, MKV, MOV, WebM or AVI with switchable subtitle tracks; M4A, MP3, WAV, FLAC or OGG audio;
  SRT, VTT and ASS subtitles; a JSON receipt of what every provider call cost.
- **Names**: known films and series use the community's spellings from the public
  [Vocnu names library](https://vocnu.com/names) (open data, CC BY-SA 4.0), so every episode spells them alike.
- **35 target languages**: `vocnu-engine languages`.

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you) and [ffmpeg](https://ffmpeg.org/) on your
  `PATH`. Tesseract is optional (image subtitles).
- An NVIDIA GPU makes speech recognition many times faster: install with `uv sync --extra cuda`. The CPU works
  too, slowly.
- Your own provider keys: `OPENAI_API_KEY` (translation) and `ELEVENLABS_API_KEY` (voices). Set them in the
  environment or in a private file such as `~/.config/vocnu/keys.env` (mode 0600, `KEY=value` lines).

## Usage

```sh
vocnu-engine translate film.mkv --to lt                 # video with voice-over and subtitles, next to the file
vocnu-engine translate talk.mp3 --to de --output audio  # voice-over audio only
vocnu-engine translate film.mp4 --to es --output subtitles
vocnu-engine translate film.mp4 --to fr --voice male --format mkv --burn-subtitles --budget 5
vocnu-engine translate film.mp4 --to lt --format mp4,mkv  # the video as an MP4 and as an MKV
vocnu-engine probe film.mkv                              # length and tracks
vocnu-engine languages
```

| Option | Meaning |
| --- | --- |
| `--to` | target language code (required) |
| `--from` | spoken language, if detection gets it wrong |
| `--output` | `video` (default for videos), `audio` or `subtitles` |
| `--voice` | `female` (default) or `male` |
| `--format`, `--audio-format` | video and audio file formats; `--format mp4,mkv` makes the video in each |
| `--burn-subtitles` | subtitles drawn into the picture |
| `--budget` | hard limit in USD on what your keys may spend for this file (default: $0.15 per minute) |
| `--out` | folder for the results (default: `<file> (vocnu <lang>)` next to it) |
| `--no-names-library` | do not read the public names library |

You pay your providers directly: in our runs about $0.05–0.10 per minute of speech for translation and voices,
and `receipt.json` shows the exact amounts. If a run stops (budget, network), run the same command again: speech
that was already made is reused, not bought twice.

Translation uses the OpenAI Responses API. `OPENAI_BASE_URL` points it at a compatible server and
`VOCNU_TRANSLATION_MODEL` picks another model; the prompts are tuned for Vocnu's default model, so quality
elsewhere may differ.

Voices are ElevenLabs premade voices (female Matilda, male George) and, for 15 languages, native narrators from
the ElevenLabs Voice Library. `VOCNU_NARRATOR_<LANGUAGE>=<voice id>` (for example `VOCNU_NARRATOR_LT`) has a voice
of your own ElevenLabs account read every line in that language, whichever `--voice` you ask for.

### Docker

```sh
docker build -t vocnu-engine .
docker run --rm -v "$PWD:/media" -v vocnu-cache:/cache -e OPENAI_API_KEY -e ELEVENLABS_API_KEY \
  vocnu-engine translate film.mkv --to lt
```

## Use it responsibly

- Translate only media you have the right to use.
- Every output carries a machine-readable note that its voice-over is AI-generated (EU AI Act, article 50), and
  the voice track is named "AI voice-over". Keep that disclosure when you publish.
- The engine uses premade voices. It does not clone anyone's voice; do not add voices of people who have not
  agreed to it.

## Development

```sh
uv sync
uv run pytest        # offline tests: no network, GPU or paid APIs
```

`vocnu_engine/` holds the whole pipeline; `pipeline.py` runs one job end to end and the module docstrings explain
each step. Language-specific guidance for translators lives in `languages.py`: notes from native speakers are
among the most valuable contributions. Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

The engine is licensed under the [GNU Affero General Public License v3.0](LICENSE): you may use, change and share
it, and if you offer it as a network service, you share your changes under the same license. For other terms (for
example, embedding it in a closed product), write to support@vocnu.com.

The Vocnu name and logo are trademarks of JAD Tech MB and are not covered by the license.
