# Contributing

Thank you for helping. Good first contributions:

- **Language notes** in `vocnu_engine/languages.py`: what a translator into your language must know (register,
  names, word order, phrases that sound translated). Show a line before and after your note.
- **Bugs** with a short clip or a receipt (`receipt.json`) that shows the problem. Never attach media you may not
  share.
- **Providers**: other speech, translation or voice services behind the same functions (`asr.py`,
  `translate.py`, `tts.py`). Open an issue first so we agree on the shape.

## How

1. Fork, branch, change, and run `uv run pytest`. New behaviour comes with a test; tests never call paid APIs.
2. Keep the style of the surrounding code: small functions, comments that explain why.
3. Open a pull request. The CLA bot asks you once to sign the [Contributor License Agreement](CLA.md) with a
   comment on the pull request. It lets JAD Tech MB keep offering Vocnu under this license and others; you keep
   the copyright of your work.

Security problems: see [SECURITY.md](SECURITY.md), not public issues.
