from pathlib import Path


with open(Path(__file__).parent / "version.py") as _version_file:
    __version__ = _version_file.read().strip()

ROOT_DIR = Path(__file__).resolve().parent