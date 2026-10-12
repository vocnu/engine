"""Settings and secrets: environment variables first, then private ~/.config/vocnu/*.env files for local runs."""
import os
from pathlib import Path
import stat

CONFIG = Path(os.environ.get('VOCNU_CONFIG_DIR', '~/.config/vocnu')).expanduser()


def env_file(path):
    """KEY=value lines from a 0600 file owned by this user; read as data, never sourced."""
    path = Path(path)
    if not path.exists():
        return {}
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError(f'Credential file must be private (0600): {path}')
        values = {}
        for line in handle:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"\'')
        return values


def setting(name):
    """An optional setting from the environment or the private files; None when it is not set."""
    value = os.environ.get(name)
    if not value and CONFIG.is_dir():
        for path in sorted(CONFIG.glob('*.env')):
            value = env_file(path).get(name)
            if value:
                break
    return value.strip() if value and value.strip() else None


def secret(name):
    value = setting(name)
    if not value or any(c.isspace() for c in value):
        raise RuntimeError(f'Missing {name}: set it in the environment or a private file in {CONFIG}')
    return value
